package com.worldsense.kaggle

import java.nio.file.Files

import com.holdenkarau.spark.testing.DataFrameSuiteBase
import com.intel.analytics.bigdl.utils.Engine
import org.apache.spark.SparkConf
import org.scalatest.FlatSpec

class LstmTest extends FlatSpec with DataFrameSuiteBase {
  override def conf: SparkConf = Engine.createSparkConf(super.conf.setMaster("local[1]").setIfMissing("spark.sql.warehouse.dir", Files.createTempDirectory("spark-warehouse").toString))
  val features = (0 until 91).map { i =>
    val sentence = if (i % 2 > 0) Seq(Seq(7.0f, 9.0f, 4.0f, 2.0f, 2.0f)) else Seq(Seq(3.0f, 5.0f, 8.0f, 2.0f, 2.0f))
    val label = if (i % 2 > 0) 1.0 else 2.0
    (sentence, label)
  }
  "Lstm" should "transform input" in {
    Engine.init
    import spark.implicits._
    assert(features.flatMap(_._1.map(_.length)).distinct.length <= 1)
    val batchSize = 4
    val embeddingDimension = features.flatMap(_._1.headOption).headOption.map(_.length).getOrElse(0)
    val paddingLength = features.map(_._1).headOption.map(_.length).getOrElse(0)
    // We must flatten sentences before feeding, because DLEstimator is not general enough for LSTM
    // TODO(davi) write a flatten transformer
    val flatFeatures = features.map(x => (x._1.flatten, x._2))
    val ds = spark.createDataset(flatFeatures)
    val df = ds.toDF("vectors", "label")
    val estimator = new Lstm()
      .setFeaturesCol("vectors")
      .setLabelCol("label")
      .setPredictionCol("p")
      .setEmbeddingDim(embeddingDimension)
      .setPaddingLength(paddingLength)
      .setBatchSize(batchSize)
    val model = estimator.fit(df)
    val p = model.transform(df)
    p.show
    assert(p.select("p").as[Double].collect.distinct.length > 1)
  }
}
