package com.worldsense.kaggle
import org.apache.spark.ml.linalg.Vector
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
    val x = "/tmp/news20/glove.6B/glove.6B.100d.txt"
    val estimator = new GloveEstimator().setVectorsPath(x).setInputCol("value").setOutputCol("vector").setSentenceLength(4)
    val model = estimator.fit(tokens)
    val vectors = model.transform(tokens).collect()
    assert(vectors.length === 1)
    val sentenceVector = vectors.head.getAs[Vector]("vector")
    assert(sentenceVector.size === 4*100)
  }
}
