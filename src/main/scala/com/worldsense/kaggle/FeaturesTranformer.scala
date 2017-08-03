package com.worldsense.kaggle

/** Read the data files from http://kaggle.com/c/quora-question-pairs.
  * The contents of both train and test files are loaded in a dataframe with minimum data transformations.
  */

import org.apache.spark.ml.Transformer
import org.apache.spark.ml.param.{Param, ParamMap}
import org.apache.spark.ml.util.{DefaultParamsWritable, Identifiable}
import org.apache.spark.sql.functions.lit
import org.apache.spark.sql.types.StructType
import org.apache.spark.sql.{DataFrame, Dataset, Encoders}

case class Features(id: BigInt, qid1: BigInt, qid2: BigInt, question1: String, question2: String, isDuplicate: Boolean)

class FeaturesTransformer(override val uid: String) extends Transformer with DefaultParamsWritable {
  val trainFile = new Param[String](this, "trainFile", "The path for the train file")

  def setTrainFile(value: String): this.type = set(trainFile, value)

  def this() = this(Identifiable.randomUID("features"))

  def copy(extra: ParamMap): FeaturesTransformer = defaultCopy(extra)

  override def transformSchema(schema: StructType): StructType = Encoders.product[Features].schema

  def transform(df: Dataset[_]): DataFrame = {
    val spark = df.sparkSession
    val csvOptions = Map("header" -> "true", "escape" -> "\"")
    val trainDF = spark.read.options(csvOptions).csv($(trainFile))
    trainDF.selectExpr(
      "cast(id as int) id",
      "cast(qid1 as int) qid1",
      "cast(qid2 as int) qid2",
      "question1", "question2",
      "cast(is_duplicate as boolean) isDuplicate"
    )
  }
}
