package com.worldsense.kaggle

import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.ml.clustering.LDA
import org.apache.spark.ml.evaluation.BinaryClassificationEvaluator
import org.apache.spark.ml.feature.CountVectorizer
import org.apache.spark.ml.param.ParamMap
import org.apache.spark.ml.tuning.{CrossValidator, CrossValidatorModel, ParamGridBuilder}
import org.apache.spark.ml.{Pipeline, PipelineModel}
import org.apache.spark.sql.SparkSession

import com.worldsense.spark.SerializedSparkJobGroup

class CrossValidateJobGroup(
    pathPrefix: String, cleanDir: String, estimatorDir: String, output: String, spark: SparkSession,
    vocabSize: Array[Int] = Array(100, 1000, 10000), minDF: Array[Double] = Array(1.0, 3.0, 10.0),
    regularization: Array[Double] = Array(0.01, 0.1, 1.0),
    numTopics: Array[Int] = Array(10, 20, 50)) extends SerializedSparkJobGroup("Cross validate the pipeline", pathPrefix, Vector(cleanDir, estimatorDir), Vector(output), spark) {
  private val logger = org.log4s.getLogger
  override def jobCode(): Boolean = {
    import spark.implicits.newProductEncoder
    val cleanDS = loadDF(cleanDir).as[CleanFeaturesJobGroup.Features]
    val trainDS = cleanDS.filter(_.source == "train")
    val testDS = cleanDS.filter(_.source == "test")
    val estimator = Pipeline.load(fullPath(pathPrefix, estimatorDir))
    // It is probably better to expose those in a final class rather than inspect into the pipeline.
    val vec = estimator.getStages(2).asInstanceOf[MultiColumnPipeline].getStage.asInstanceOf[CountVectorizer]
    val lda = estimator.getStages(3).asInstanceOf[MultiColumnPipeline].getStage.asInstanceOf[LDA]
    val lr = estimator.getStages.last.asInstanceOf[LogisticRegression]
    // Grid search on hyperparameter space
    val paramGrid = new ParamGridBuilder()
        .addGrid(vec.vocabSize, vocabSize)
        .addGrid(vec.minDF, minDF)
        .addGrid(lr.regParam, regularization)
        .addGrid(lda.k, numTopics)
        .build()

    val evaluator = new BinaryClassificationEvaluator()
      .setLabelCol("isDuplicateLabel")
        .setRawPredictionCol("raw")
        .setMetricName("areaUnderROC")

    // Cross-validation setup
    val numFolds = 5
    val cv = new CrossValidator()
        .setEstimator(estimator)
        .setEvaluator(evaluator)
        .setEstimatorParamMaps(paramGrid)
        .setNumFolds(numFolds)

    // Train model. This also runs the indexers.
    val crossValidatedModel: CrossValidatorModel = cv.fit(trainDS)
    logger.info(f"Trained a cross-validated model from a training dataset of size ${trainDS.count()} " +
                f"with areaUnderROC ${crossValidatedModel.avgMetrics.max}.")

    // Get paramMap from best model to check hyperparams learned by grid
    val crossValidatedHyperParams: ParamMap =
    crossValidatedModel.getEstimatorParamMaps.zip(
      crossValidatedModel.avgMetrics).maxBy(_._2)._1
    logger.info(s"best hyperparams: $crossValidatedHyperParams")

    val pipelineModel: PipelineModel = crossValidatedModel.bestModel.asInstanceOf[PipelineModel]
    pipelineModel.write.save(fullPath(pathPrefix, output))
    true
  }
}
