package com.worldsense.kaggle

import org.apache.spark.ml.PipelineModel
import org.apache.spark.ml.linalg.DenseVector
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.functions.col

import com.worldsense.spark.SerializedSparkJobGroup
class SubmissionJobGroup(
    pathPrefix: String, cleanDir: String, modelDir: String, submissionDir: String,
    spark: SparkSession) extends SerializedSparkJobGroup(
      "Prepares a csv submission file for the kaggle quora challenge",
      pathPrefix, Vector(cleanDir, modelDir), Vector(submissionDir), spark) {
  import spark.implicits.{newIntEncoder, newProductEncoder}
  override def jobCode(): Boolean = {
    val cleanDF = loadDF(cleanDir)
    val testDF = cleanDF.filter(col("source") === "test")
    val model = PipelineModel.load(fullPath(pathPrefix, modelDir))
    val predictions = model.transform(testDF).select("p", "testId").as[(DenseVector, String)]
    val submission = predictions map { case (p, testId) =>
      (p.values.last, testId)
    } toDF("is_duplicate", "test_id")
    val csvOptions = Map("header" -> "true", "escape" -> "\"")
    submission.repartition(1).write.options(csvOptions).csv(fullPath(pathPrefix, submissionDir))
    true
  }
}
