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

import com.ibm.icu.text.{Normalizer2, Transliterator}
import org.apache.spark.ml.Transformer
import org.apache.spark.ml.param.{Param, ParamMap}
import org.apache.spark.ml.util.{DefaultParamsWritable, Identifiable}
import org.apache.spark.sql.types.StructType
import org.apache.spark.sql.{DataFrame, Dataset, Encoders}

class CleanFeaturesTransformer(override val uid: String) extends Transformer with DefaultParamsWritable {
  def this() = this(Identifiable.randomUID("cleanfeatures"))
  def copy(extra: ParamMap): CleanFeaturesTransformer = defaultCopy(extra)

  val removeDiacriticalsParam: Param[Boolean] = new Param(this, "tokenizer", "estimator for selection")
  def setTokenizer(value: Boolean): this.type = set(removeDiacriticalsParam, value)
  setDefault(removeDiacriticalsParam, true)

  override def transformSchema(schema: StructType): StructType = Encoders.product[Features].schema
  def transform(ds: Dataset[_]): DataFrame = {
    import CleanFeaturesTransformer.normalize
    import ds.sparkSession.implicits.newProductEncoder
    // We replace null cells with the most straightforward default values.
    val features = ds.na.fill(0).na.fill("").as[Features]
    val cleanFeatures = features map { row =>
      row.copy(
        question1 = normalize(row.question1, $(removeDiacriticalsParam)),
        question2 = normalize(row.question2, $(removeDiacriticalsParam))
      )
    }
    cleanFeatures.toDF
  }
}

object CleanFeaturesTransformer {
  private val normalizer = Normalizer2.getNFKCCasefoldInstance
  // Remove diacriticals, straight from http://userguide.icu-project.org/transforms/general
  private val unaccenter = Transliterator.getInstance("NFD; [:Nonspacing Mark:] Remove; NFC")
  private[kaggle] def normalize(text: String, removeDiacriticals: Boolean = true): String = {
    if (removeDiacriticals) normalizer.normalize(unaccenter.transliterate(text))
    else normalizer.normalize(text)
  }
}