package com.worldsense.kaggle
import com.worldsense.kaggle.FeaturesLoader.Features
import org.apache.spark.ml.Transformer
import org.apache.spark.ml.linalg.DenseVector
import org.apache.spark.ml.param.{Param, ParamMap}
import org.apache.spark.ml.util.{DefaultParamsWritable, Identifiable}
import org.apache.spark.sql.types.StructType
import org.apache.spark.sql.{DataFrame, Dataset, Encoders}

class SubmissionWriter(override val uid: String) extends Transformer with DefaultParamsWritable {
  private val csvOptions = Map("header" -> "true", "escape" -> "\"")

  def this() = this(Identifiable.randomUID("submission"))
  val modelParam: Param[Transformer] = new Param(this, "model", "the model")
  def setModel(value: Transformer): this.type = set(modelParam, value)
  val testFile = new Param[String](this, "testFile", "The path for the test file")
  def setTestFile(value: String): this.type = set(testFile, value)

  def copy(extra: ParamMap): SubmissionWriter = defaultCopy(extra)

  override def transformSchema(schema: StructType): StructType = Encoders.product[Features].schema

  def transform(df: Dataset[_]): DataFrame = {
    import df.sparkSession.implicits.newProductEncoder
    val predictions = $(modelParam).transform(df).select("p", "id").as[(DenseVector, String)]
    predictions map { case (p, testId) =>
      (p.values.last, testId)
    } toDF("is_duplicate", "test_id")
  }

  def writeSubmissionFile(features: Dataset[Features], submissionFilePath: String): Unit = {
    val submission = transform(features)
    submission.repartition(1).write.options(csvOptions).csv(submissionFilePath)
  }
}
