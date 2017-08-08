package com.worldsense.kaggle

import com.holdenkarau.spark.testing.DataFrameSuiteBase
import org.scalatest.FlatSpec

class GloveTest extends FlatSpec with DataFrameSuiteBase {
  val gloveFile = getClass.getResource("/glove/glove.6B.300d.sample.txt").getFile
  "GloveTest" should "load file" in {
    import spark.implicits.newStringEncoder
    val glove = new GloveEstimator().setVectorsPath(gloveFile).setInputCol("value")
    glove.fit(spark.emptyDataset[String])
  }
  it should "transform input" in {
    import spark.implicits._
    val tokens = spark.createDataset(Seq(Array("token", "supercalifragilistibonitao", "TOKEN")))
    val estimator = new GloveEstimator().setVectorsPath(gloveFile).setInputCol("value")
    val model = estimator.fit(tokens)
    val vectors = model.transform(tokens).collect()
    assert(vectors.nonEmpty)
  }
}
