package com.worldsense.kaggle

import com.worldsense.kaggle.QuoraQuestionsPairsCrossValidator.LogLossBinaryClassificationEvaluator
import org.apache.spark.ml.Estimator
import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.ml.clustering.LDA
import org.apache.spark.ml.evaluation.Evaluator
import org.apache.spark.ml.feature.{CountVectorizer, IDF, StopWordsRemover}
import org.apache.spark.ml.linalg.Vector
import org.apache.spark.ml.param.{Param, ParamMap}
import org.apache.spark.ml.tuning.{CrossValidator, CrossValidatorModel, ParamGridBuilder}
import org.apache.spark.ml.util.{DefaultParamsReadable, DefaultParamsWritable, Identifiable}
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.functions.col
import org.apache.spark.sql.types.{DoubleType, StructType}
import org.apache.spark.sql.{Dataset, Row}

import scala.io.Source

class QuoraQuestionsPairsCrossValidator(override val uid: String) extends Estimator[CrossValidatorModel] {
  def this() = this(Identifiable.randomUID("quoraquestionspairscrossvalidator"))
  val numFolds = 3

  final val stopwords: Param[List[String]] = new Param[List[String]](this, "stopwords", "path to files with comma separate normalized stopwords")
  def setStopwords(value: List[String]): this.type = set(stopwords, value)
  setDefault(stopwords, List("src/main/resources/quora/stopwords.txt"))

  final val minDocFreq: Param[List[Int]] = new Param[List[Int]](this, "minDF", "comma separate input column names")
  def setMinDocFreq(value: List[Int]): this.type = set(minDocFreq, value)
  setDefault(minDocFreq, List(2, 5))

  final val numTopics: Param[List[Int]] = new Param[List[Int]](this, "numTopics", "comma separate input column names")
  def setNumTopics(value: List[Int]): this.type = set(numTopics, value)
  setDefault(numTopics, List(20, 50, 100))

  final val ldaMaxIter: Param[List[Int]] = new Param[List[Int]](this, "ldaMaxIter", "comma separate input column names")
  def setLdaMaxIter(value: List[Int]): this.type = set(ldaMaxIter, value)
  setDefault(ldaMaxIter, List(100, 1000))

  final val logisticRegressionMaxIter: Param[List[Int]] = new Param[List[Int]](this, "logisticRegressionMaxIter", "comma separate input column names")
  def setLogisticRegressionMaxIter(value: List[Int]): this.type = set(logisticRegressionMaxIter, value)
  setDefault(logisticRegressionMaxIter, List(100, 1000))

  override def transformSchema(schema: StructType): StructType = assembleCrossValidator().transformSchema(schema)
  override def fit(dataset: Dataset[_]): CrossValidatorModel = {
    assembleCrossValidator().fit(dataset)
  }
  def copy(extra: ParamMap): QuoraQuestionsPairsCrossValidator = defaultCopy(extra)

  private def assembleCrossValidator(): CrossValidator = {
    val stopwordsRemover = new StopWordsRemover()
    val idf = new IDF()
    val countVectorizer = new CountVectorizer()
    val logisticRegression = new LogisticRegression()
    val lda = new LDA()
    val estimator = new QuoraQuestionsPairsPipeline()
      .setStopwordsRemover(stopwordsRemover)
      .setIDF(idf)
      .setLDA(lda)
      .setLogisticRegression(logisticRegression)
    // Grid search on hyperparameter space
    val stopwordsLists = $(stopwords).map(Source.fromFile).map(_.getLines().mkString(",").split(","))
    val paramGrid = new ParamGridBuilder()
      .addGrid(stopwordsRemover.stopWords, stopwordsLists)
      .addGrid(idf.minDocFreq, $(minDocFreq))
      .addGrid(lda.k, $(numTopics))
      .addGrid(lda.maxIter, $(ldaMaxIter))
      .addGrid(logisticRegression.maxIter, $(logisticRegressionMaxIter))
      .build()

    val evaluator = new LogLossBinaryClassificationEvaluator()
      .setLabelCol("isDuplicateLabel").setProbabilityCol("p")

    // Cross-validation setup
    new CrossValidator()
        .setEstimator(estimator)
        .setEvaluator(evaluator)
        .setEstimatorParamMaps(paramGrid)
        .setNumFolds(numFolds)
  }
}

object QuoraQuestionsPairsCrossValidator {
  private def logLoss(scoreAndLabels: RDD[(Double, Double)]): Double = {
    val rows = scoreAndLabels.collect
    // https://www.kaggle.com/wiki/LogLoss
    val v = rows.map(pl => pl._2 * math.log(pl._1)).sum / rows.length * -1
    logger.info(s"Computed log loss of $v for ${rows.length} rows.")
    v
  }
  private val logger = org.log4s.getLogger

  class LogLossBinaryClassificationEvaluator(override val uid: String) extends Evaluator with DefaultParamsWritable {
    def this() = this(Identifiable.randomUID("logLossEval"))
    override def copy(extra: ParamMap): LogLossBinaryClassificationEvaluator = defaultCopy(extra)
    final val probabilityCol: Param[String] = new Param[String](this, "probabilityCol", "Column name for predicted class conditional probabilities.")
    setDefault(probabilityCol, "probability")
    final def getProbabilityCol: String = $(probabilityCol)
    final def setProbabilityCol(value: String): this.type = set(probabilityCol, value)
    final val labelCol: Param[String] = new Param[String](this, "labelCol", "label column name")
    setDefault(labelCol, "label")
    final def getLabelCol: String = $(labelCol)
    final def setLabelCol(value: String): this.type = set(labelCol, value)

    override def evaluate(dataset: Dataset[_]): Double = {
      val scoreAndLabels: RDD[(Double, Double)] =
        dataset.select(col($(probabilityCol)), col($(labelCol)).cast(DoubleType)).rdd.map {
          case Row(probability: Vector, label: Double) => (probability(1), label)
          case Row(probability: Double, label: Double) => (probability, label)
        }
      logLoss(scoreAndLabels)
    }
  }
  object LogLossBinaryClassificationEvaluator extends DefaultParamsReadable[LogLossBinaryClassificationEvaluator]
}
