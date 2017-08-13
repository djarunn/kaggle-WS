package com.worldsense.kaggle

import com.holdenkarau.spark.testing.DataFrameSuiteBase
import com.intel.analytics.bigdl.utils.Engine
import org.apache.spark.SparkConf
import org.scalatest.FlatSpec

class LstmTest extends FlatSpec with DataFrameSuiteBase {
  override def conf: SparkConf = Engine.createSparkConf(super.conf.setMaster("local[1]"))
  "Lstm" should "transform input" in {
    Engine.init
    import spark.implicits._
    val sentenceVector = Seq(0.11f, 0.21f, 0.51f, 0.61f)
    val sentenceLabel = Seq(1.0, 0.0f)
    val ds = spark.createDataset(Seq((sentenceVector, sentenceLabel)))
    val df = ds.toDF("vectors", "label")
    val estimator = new Lstm()
      .setFeaturesCol("vectors")
      .setLabelCol("label")
      .setPredictionCol("p")
      .setEmbeddingDim(4)
    val model = estimator.fit(df)
    val p = model.transform(df).collect()
    assert(p.nonEmpty)
  }
}
