package com.worldsense.kaggle

import com.github.fommil.netlib.BLAS
import net.sourceforge.argparse4j.ArgumentParsers
import net.sourceforge.argparse4j.inf.ArgumentParserException
import org.apache.spark.sql.SparkSession

import scala.util.{Failure, Try}

object Main extends App {
  private val logger = org.log4s.getLogger
  private val spark = SparkSession.builder.master("local").appName("kaggle").getOrCreate()
  private val featuresLoader = new FeaturesLoader()
  private val parser = ArgumentParsers.newArgumentParser("kaggle")
    .description("Train and evaluate a model for kaggle's  quora questions pair challenge.")
  parser.addArgument("trainingDataFile")
    .help("The training data file provided by kaggle")
  parser.addArgument("testDataFile")
    .help("The test data file provided by kaggle")
  parser.addArgument("submissionFile")
    .help("The file to create with the results")
  Try(parser.parseArgs(args)) recoverWith { case e: ArgumentParserException =>
    parser.handleError(e)
    System.exit(1)
    Failure(e)
  } foreach { res =>
    logger.info(s"BLAS backend is ${BLAS.getInstance().getClass.getName}")
    run(res.getString("trainingDataFile"), res.getString("testDataFile"), res.getString("submissionFile"))
  }
  def run(trainingDataFile: String, testDataFile: String, submissionFile: String): Unit = {
    val estimator = new QuoraQuestionsPairsCrossValidator
    logger.info(s"Cross validator params:\n${estimator.explainParams()}")
    val numVariations = estimator.extractParamMap().toSeq.map(_.value.asInstanceOf[List[_]].length).product
    logger.info(s"Cross validator will train $numVariations * ${estimator.numFolds} models")

    val trainData = featuresLoader.loadTrainFile(spark, trainingDataFile)
    val model = estimator.fit(trainData)

    val testData = featuresLoader.loadTestFile(spark, testDataFile)
    val submissionWriter = new SubmissionWriter().setModel(model)
    submissionWriter.writeSubmissionFile(testData, submissionFile)
  }
}
