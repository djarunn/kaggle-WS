package com.worldsense.kaggle

/** Consumes a dataframe from RawFeaturesJobGroup and cleans the data on it.
 *
 * Besides simple data transformations and empty value handling, this cleans the text using
 * worldsense bow machinery, then rewrites the cleaner version of the
 * text as CleanFeaturesJobGroup.Features, which can be easily processed by subsequent pipelines.
 *
 * The cleaning strategy simply delegates to existing worldsense pipeline, which will parse the
 * questions as if they were HTML documents, generate BoW representations, and then assemble
 * then again.
 */

import org.apache.spark.ml.Transformer
import org.apache.spark.ml.param.ParamMap
import org.apache.spark.ml.util.{DefaultParamsWritable, Identifiable}
import org.apache.spark.sql.types.StructType
import org.apache.spark.sql.{DataFrame, Dataset, Encoders}

class CleanFeaturesTransformer(override val uid: String) extends Transformer with DefaultParamsWritable {
  def this() = this(Identifiable.randomUID("cleanfeatures"))
  def copy(extra: ParamMap): FeaturesTransformer = defaultCopy(extra)

  override def transformSchema(schema: StructType): StructType = Encoders.product[Features].schema
  def transform(ds: Dataset[_]): DataFrame = {
    import ds.sparkSession.implicits.newProductEncoder
    val features = ds.as[Features]
    val cleanFeatures = features map { row =>
      row.copy(
        question1 = CleanFeaturesTransformer.cleanQuestion(row.qid1, row.question1),
        question2 = CleanFeaturesTransformer.cleanQuestion(row.qid2, row.question2))
    }
    cleanFeatures.toDF
  }
}

object CleanFeaturesTransformer {
  private[kaggle] def cleanQuestion(qid: BigInt, question: String): String = {
    question
  }
}