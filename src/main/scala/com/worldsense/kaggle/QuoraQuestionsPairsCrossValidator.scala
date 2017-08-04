package com.worldsense.kaggle

import org.apache.spark.ml.Estimator
import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.ml.clustering.LDA
import org.apache.spark.ml.evaluation.BinaryClassificationEvaluator
import org.apache.spark.ml.feature.CountVectorizer
import org.apache.spark.ml.param.{Param, ParamMap}
import org.apache.spark.ml.tuning.{CrossValidator, CrossValidatorModel, ParamGridBuilder}
import org.apache.spark.ml.util.Identifiable
import org.apache.spark.sql.Dataset
import org.apache.spark.sql.types.StructType

class QuoraQuestionsPairsCrossValidator(override val uid: String) extends Estimator[CrossValidatorModel] {
  def this() = this(Identifiable.randomUID("quoraquestionspairscrossvalidator"))
  val numFolds = 3
  private val logger = org.log4s.getLogger

  final val vocabularySize: Param[List[Int]] = new Param[List[Int]](this, "vocabularySize", "comma separate input column names")
  def setVocabularySize(value: List[Int]): this.type = set(vocabularySize, value)
  setDefault(vocabularySize, List(10000,1000000))

  final val regularization: Param[List[Double]] = new Param[List[Double]](this, "regularization", "comma separate input column names")
  def setRegularization(value: List[Double]): this.type = set(regularization, value)
  setDefault(regularization, List(0.01,0.1))

  final val numTopics: Param[List[Int]] = new Param[List[Int]](this, "numTopics", "comma separate input column names")
  def setNumTopics(value: List[Int]): this.type = set(numTopics, value)
  setDefault(numTopics, List(20,50))

  final val minDF: Param[List[Double]] = new Param[List[Double]](this, "minDF", "comma separate input column names")
  def setMinDF(value: List[Double]): this.type = set(minDF, value)
  setDefault(minDF, List(3.0))

  final val ldaMaxIter: Param[List[Int]] = new Param[List[Int]](this, "ldaMaxIter", "comma separate input column names")
  def setLdaMaxIter(value: List[Int]): this.type = set(ldaMaxIter, value)
  setDefault(ldaMaxIter, List(20, 100))

  final val logisticRegressionMaxIter: Param[List[Int]] = new Param[List[Int]](this, "logisticRegressionMaxIter", "comma separate input column names")
  def setLogisticRegressionMaxIter(value: List[Int]): this.type = set(logisticRegressionMaxIter, value)
  setDefault(logisticRegressionMaxIter, List(100))

  override def transformSchema(schema: StructType): StructType = assembleCrossValidator().transformSchema(schema)
  override def fit(dataset: Dataset[_]): CrossValidatorModel = {
    assembleCrossValidator().fit(dataset)
  }
  def copy(extra: ParamMap): QuoraQuestionsPairsCrossValidator = defaultCopy(extra)

  private def assembleCrossValidator(): CrossValidator = {
    val countVectorizer = new CountVectorizer()
    val logisticRegression = new LogisticRegression()
    val lda = new LDA()
    val estimator = new QuoraQuestionsPairsPipeline()
      .setCountVectorizer(countVectorizer)
      .setLogisticRegression(logisticRegression)
      .setLDA(lda)
    // Grid search on hyperparameter space
    val paramGrid = new ParamGridBuilder()
      .addGrid(countVectorizer.vocabSize, $(vocabularySize))
      .addGrid(countVectorizer.minDF, $(minDF))
      .addGrid(logisticRegression.regParam, $(regularization))
      .addGrid(lda.k, $(numTopics))
      .build()

    val evaluator = new BinaryClassificationEvaluator()
      .setLabelCol("isDuplicateLabel")
        .setRawPredictionCol("raw")
        .setMetricName("areaUnderROC")

    // Cross-validation setup
    new CrossValidator()
        .setEstimator(estimator)
        .setEvaluator(evaluator)
        .setEstimatorParamMaps(paramGrid)
        .setNumFolds(numFolds)
  }
}
