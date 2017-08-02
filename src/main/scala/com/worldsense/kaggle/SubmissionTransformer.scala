package com.worldsense.kaggle

import org.apache.spark.ml.{PipelineModel, Transformer}
import org.apache.spark.ml.linalg.DenseVector
import org.apache.spark.ml.param.{Param, ParamMap}
import org.apache.spark.ml.util.{DefaultParamsWritable, Identifiable}
import org.apache.spark.sql.{DataFrame, Dataset, Encoders}
import org.apache.spark.sql.types.StructType

class SubmissionTransformer(override val uid: String) extends Transformer with DefaultParamsWritable {
  def this() = this(Identifiable.randomUID("submission"))
  val modelParam: Param[Transformer] = new Param(this, "model", "the model")
  def setModel(value: Transformer): this.type = set(modelParam, value)
  val testFile = new Param[String](this, "testFile", "The path for the test file")
  def setTestFile(value: String): this.type = set(testFile, value)
  val outputFile = new Param[String](this, "testFile", "The path for the test file")
  def setOutputFile(value: String): this.type = set(outputFile, value)

  def copy(extra: ParamMap): SubmissionTransformer = defaultCopy(extra)

  override def transformSchema(schema: StructType): StructType = Encoders.product[Features].schema

  def transform(df: Dataset[_]): DataFrame = {
    import df.sparkSession.implicits.newProductEncoder
    val csvOptions = Map("header" -> "true", "escape" -> "\"")
    val testDF = df.sparkSession.read.options(csvOptions).csv($(testFile))
    val predictions = $(modelParam).transform(testDF).select("p", "testId").as[(DenseVector, String)]
    val submission = predictions map { case (p, testId) =>
      (p.values.last, testId)
    } toDF("is_duplicate", "test_id")
    submission.repartition(1).write.options(csvOptions).csv($(outputFile))
    submission
  }
}
