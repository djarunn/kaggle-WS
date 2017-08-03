package com.worldsense.kaggle

import java.nio.file.{Files, Path}

import com.holdenkarau.spark.testing.DataFrameSuiteBase
import org.scalatest.FlatSpec

class FeaturesTransformerTest extends FlatSpec with DataFrameSuiteBase {
  "FeaturesTransformerTest" should "serialize back and forth" in {
    import spark.implicits.newProductEncoder
    val tmpdir: Path = Files.createTempDirectory("featurestransformertest")
    val trainFile = getClass.getResource("/quora/train100.csv").getFile
    val featuresTransformer = new FeaturesTransformer()
      .setTrainFile(trainFile)
    val features = featuresTransformer.transform(spark.emptyDataFrame).as[Features].collect()
    assert(features.exists(row => Option(row.question2).isEmpty))  // nulls have not been cleaned
    assert(features.length === 100)
  }
}
