package com.worldsense.kaggle

import org.apache.spark.ml.Estimator
import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.ml.clustering.LDA
import org.apache.spark.ml.evaluation.BinaryClassificationEvaluator
import org.apache.spark.ml.feature.CountVectorizer
import org.apache.spark.ml.param.{Param, ParamMap}
import org.apache.spark.ml.tuning.{CrossValidator, CrossValidatorModel, ParamGridBuilder}
import org.apache.spark.ml.util.{DefaultParamsReadable, DefaultParamsWritable, Identifiable}
import org.apache.spark.sql.Dataset
import org.apache.spark.sql.types.StructType

class QuoraQuestionsPairsCrossValidator(override val uid: String) extends Estimator[CrossValidatorModel] with DefaultParamsWritable {
  def this() = this(Identifiable.randomUID("quoraquestionspairscrossvalidator"))
  private val logger = org.log4s.getLogger
    final val vocabularySize: Param[Array[Int]] = new Param[Array[Int]](this, "vocabularySize", "comma separate input column names")
  setDefault(vocabularySize, Array(1000,1000000))
  final val regularization: Param[Array[Double]] = new Param[Array[Double]](this, "regularization", "comma separate input column names")
  setDefault(regularization, Array(0.01,0.1,1.0))
  final val numTopics: Param[Array[Int]] = new Param[Array[Int]](this, "numTopics", "comma separate input column names")
  setDefault(numTopics, Array(10,20,50))
  final val minDF: Param[Array[Double]] = new Param[Array[Double]](this, "minDF", "comma separate input column names")
  setDefault(minDF, Array(3.0))
  final val ldaMaxIterations: Param[Array[Int]] = new Param[Array[Int]](this, "ldaMaxIterations", "comma separate input column names")
  setDefault(ldaMaxIterations, Array(3))
  final val lrMaxIterations: Param[Array[Int]] = new Param[Array[Int]](this, "ldaMaxIterations", "comma separate input column names")
  setDefault(lrMaxIterations, Array(3))

  override def transformSchema(schema: StructType): StructType = assembleCrossValidator().transformSchema(schema)
  override def fit(dataset: Dataset[_]): CrossValidatorModel = assembleCrossValidator().fit(dataset)
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
    val numFolds = 5
    new CrossValidator()
        .setEstimator(estimator)
        .setEvaluator(evaluator)
        .setEstimatorParamMaps(paramGrid)
        .setNumFolds(numFolds)
  }
}

object QuoraQuestionsPairsCrossValidator extends DefaultParamsReadable[QuoraQuestionsPairsCrossValidator]
