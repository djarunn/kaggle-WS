package com.worldsense.kaggle

import org.apache.spark.ml.evaluation.BinaryClassificationEvaluator
import org.apache.spark.ml.param.Param
import org.apache.spark.ml.tuning.{CrossValidator, ParamGridBuilder}

class QuoraQuestionsPairsCrossValidator extends CrossValidator {
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

  private def assembleCrossValidator = {
    val estimator = new QuoraQuestionsPairsPipeline()
    // Grid search on hyperparameter space
    val paramGrid = new ParamGridBuilder()
      .addGrid(estimator.countVectorizer.vocabSize, $(vocabularySize))
      .addGrid(estimator.countVectorizer.minDF, $(minDF))
      .addGrid(estimator.logisticRegression.regParam, $(regularization))
      .addGrid(estimator.latentDirichletAllocator.k, $(numTopics))
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
