package com.worldsense.kaggle

import com.holdenkarau.spark.testing.DataFrameSuiteBase
import org.scalatest.FlatSpec

class CleanFeaturesTransformerTest extends FlatSpec with DataFrameSuiteBase {
  val features = Array(
    Features(id = 0, qid1 = 1, qid2 = null,
      question1 = "joao", question2 = "joão",
      isDuplicate = false)
  )
  "CleanFeaturesTransformerTest" should "get rid of nulls and diacriticals" in {
    import spark.implicits._
    val transformer = new CleanFeaturesTransformer()
    val ds = spark.createDataset(features)
    val clean = transformer.transform(ds).as[Features].collect().head
    assert(Option(clean.qid2).exists(_ === 0))  // filled null
    assert(clean.question1 === clean.question2)  // dropped diacriticals
  }
}