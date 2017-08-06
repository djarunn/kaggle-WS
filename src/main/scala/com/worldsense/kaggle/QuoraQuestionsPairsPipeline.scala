package com.worldsense.kaggle

import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.ml.clustering.LDA
import org.apache.spark.ml.feature._
import org.apache.spark.ml.linalg.DenseVector
import org.apache.spark.ml.param.{Param, ParamMap}
import org.apache.spark.ml.util.Identifiable
import org.apache.spark.ml.{Estimator, Pipeline, PipelineModel, PipelineStage}
import org.apache.spark.mllib.evaluation.BinaryClassificationMetrics
import org.apache.spark.sql.Dataset
import org.apache.spark.sql.types.StructType


class QuoraQuestionsPairsPipeline(override val uid: String) extends Estimator[PipelineModel] {
  val cleanFeaturesTransformer: Param[CleanFeaturesTransformer] =
    new Param(this, "cleaner", "Cleans the input text.")
  setDefault(cleanFeaturesTransformer, new CleanFeaturesTransformer())
  val tokenizer: Param[Tokenizer] =
    new Param(this, "tokenizer", "Breaks input text into tokens.")
  setDefault(tokenizer, new Tokenizer)
  val stopwordsRemover: Param[StopWordsRemover] =
    new Param(this, "stopwords", "Drops stopwords from input text.")
  setDefault(stopwordsRemover, new StopWordsRemover())
  val countVectorizer: Param[CountVectorizer] =
    new Param(this, "countVectorizer", "Convert input tokens into weighted vectors.")
  setDefault(countVectorizer, new CountVectorizer())
  val lda: Param[LDA] =
    new Param(this, "lda", "Convert each question into a weighted topic vector.")
  setDefault(lda, new LDA())
  val logisticRegression: Param[LogisticRegression] =
    new Param(this, "logisticRegression", "Combine question vectors pairs into a predicted probability.")
  setDefault(logisticRegression, new LogisticRegression())
  private val questionsCols = Array("question1", "question2")

  def this() = this(Identifiable.randomUID("quoraquestionspairspipeline"))

  override def transformSchema(schema: StructType): StructType = assemblePipeline().transformSchema(schema)

  private val logger = org.log4s.getLogger
  private def assemblePipeline(): Pipeline = {
    val stages = Array(
      cleanerPipeline(),
      tokenizePipeline(),
      vectorizePipeline(),
      ldaPipeline(),
      logisticRegressionPipeline()
    ).flatten
    new Pipeline().setStages(stages)
  }
  override def fit(dataset: Dataset[_]): PipelineModel = {
    import dataset.sparkSession.implicits.newProductEncoder
    logger.info(s"Preparing to fit quora question pipeline with params:\n${explainParams()}")
    val model = assemblePipeline().fit(dataset)
    val predictions = model.transform(dataset).select("p", "isDuplicateLabel").as[(DenseVector, Int)]
    val predictionsAndLabels = predictions map { case (p, label) =>
      (p.values.last, label.toDouble)
    }
    val metrics = new BinaryClassificationMetrics(predictionsAndLabels.rdd)
    val areaUnderPR = metrics.areaUnderPR()
    val areaUnderROC = metrics.areaUnderROC()
    logger.info(s"Trained a model with area under pr $areaUnderPR and area under roc curve $areaUnderROC")
    model
  }

  override def explainParams(): String = {
    val stages = Seq(
      $(cleanFeaturesTransformer), $(tokenizer), $(stopwordsRemover), $(countVectorizer), $(lda), $(logisticRegression)
    )
    stages.map(_.explainParams()).mkString("\n")
  }

  private def cleanerPipeline(): Array[PipelineStage] = {
    Array($(cleanFeaturesTransformer))
  }

  private def tokenizePipeline(): Array[PipelineStage] = {
    val mcTokenizer = new MultiColumnPipeline()
        .setStage($(tokenizer))
        .setInputCols(questions(""))
        .setOutputCols(questions("all_tokens"))
    val mcStopwordsRemover = new MultiColumnPipeline()
        .setStage($(stopwordsRemover))
        .setInputCols(mcTokenizer.getOutputCols)
        .setOutputCols(questions("tokens"))
    Array(mcTokenizer, mcStopwordsRemover)
  }

  private def vectorizePipeline(): Array[PipelineStage] = {
    val mcCountVectorizer = new MultiColumnPipeline()
      .setInputCols(questions("tokens"))
      .setOutputCols(questions("vectors"))
      .setStage($(countVectorizer))
    Array(mcCountVectorizer)
  }

 private def ldaPipeline(): Array[PipelineStage] = {
   // The "em" optimizer is distributed, supports serialization, but is disk hungry and slow.
   // The "online" runs in the driver, is fast but cannot be serialized.
   // We use the latter, since this model is only used to create a submission and nothing else.
   val optimizer = "online"
   val ldaEstimator = $(lda)
     .setOptimizer(optimizer)
     .setFeaturesCol("tmpinput").setTopicDistributionCol("tmpoutput")
   val mcLda = new MultiColumnPipeline()
     .setInputCols(questions("vectors"))
     .setOutputCols(questions("lda"))
     .setStage(ldaEstimator, ldaEstimator.getFeaturesCol, ldaEstimator.getTopicDistributionCol)
   Array(mcLda)
  }

  private def questions(suffix: String) = questionsCols.map(_ + suffix)

  private def logisticRegressionPipeline(): Array[PipelineStage] = {
    val labelCol = "isDuplicateLabel"
    val assembler = new VectorAssembler().setInputCols(questions("lda")).setOutputCol("mergedlda")
    val labeler = new SQLTransformer().setStatement(
      s"SELECT *, cast(isDuplicate as int) $labelCol from __THIS__")
    // See https://www.kaggle.com/davidthaler/how-many-1-s-are-in-the-public-lb
    val weight = new SQLTransformer().setStatement(
      s"SELECT *, IF(isDuplicate, 0.47, 1.3) lrw from __THIS__")
    val lr = $(logisticRegression)
        .setFeaturesCol("mergedlda").setProbabilityCol("p").setRawPredictionCol("raw")
        .setWeightCol("lrw")
        .setLabelCol(labelCol)
    Array(assembler, labeler, weight, lr)
  }

  def copy(extra: ParamMap): QuoraQuestionsPairsPipeline = defaultCopy(extra)

  def setCleanFeaturesTransformer(value: CleanFeaturesTransformer): this.type = set(cleanFeaturesTransformer, value)

  def setTokenizer(value: Tokenizer): this.type = set(tokenizer, value)

  def setStopwordsRemover(value: StopWordsRemover): this.type = set(stopwordsRemover, value)

  def setCountVectorizer(value: CountVectorizer): this.type = set(countVectorizer, value)

  def setLDA(value: LDA): this.type = set(lda, value)

  def setLogisticRegression(value: LogisticRegression): this.type = set(logisticRegression, value)
}
