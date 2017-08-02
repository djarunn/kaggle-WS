package com.worldsense.kaggle

import java.io.File

import org.apache.commons.io.FileUtils
import org.junit.runner.RunWith
import org.scalatest.junit.JUnitRunner

// scalastyle:off magic.number
/*
@RunWith(classOf[JUnitRunner])
class CleanFeaturesJobGroupTest extends JobGroupSpec {
  import spark.implicits._
  val rawResource = new File(getClass.getResource("/quora/raw.json").getFile)
  val raw = new File(HadoopUtils.appendPath(datadir, rawResource.getName))
  val clean = HadoopUtils.appendPath(datadir, "clean")
  def populateDatadir() = {
    FileUtils.copyFile(rawResource, raw)
  }
  "CleanFeaturesJobGroup" should "should create a typed dataframe with all rows" in {
    populateDatadir()
    new CleanFeaturesJobGroup(datadir, raw.getName, "clean", spark).run()
    val cleanFeatures = spark.read.parquet(clean).as[CleanFeaturesJobGroup.Features]
    // There are 100 test rows and 100 train rows in raw.json.
    assert(cleanFeatures.count() === 2*100)
  }
}
*/
