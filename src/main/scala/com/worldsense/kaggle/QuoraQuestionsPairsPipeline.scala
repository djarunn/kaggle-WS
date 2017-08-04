package com.worldsense.kaggle

import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.ml.clustering.LDA
import org.apache.spark.ml.feature._
import org.apache.spark.ml.param.{Param, ParamMap}
import org.apache.spark.ml.util.Identifiable
import org.apache.spark.ml.{Estimator, Pipeline, PipelineModel, PipelineStage}
import org.apache.spark.sql.Dataset
import org.apache.spark.sql.types.StructType


class QuoraQuestionsPairsPipeline(override val uid: String) extends Estimator[PipelineModel] {
  val cleanFeaturesTransformerParam: Param[CleanFeaturesTransformer] =
    new Param(this, "cleaner", "Cleans the input text.")
  setDefault(cleanFeaturesTransformerParam, new CleanFeaturesTransformer())
  val tokenizerParam: Param[Tokenizer] =
    new Param(this, "tokenizer", "Breaks input text into tokens.")
  setDefault(tokenizerParam, new Tokenizer)
  val stopwordsRemoverParam: Param[StopWordsRemover] =
    new Param(this, "stopwords", "Drops stopwords from input text.")
  setDefault(stopwordsRemoverParam, new StopWordsRemover())
  val countVectorizerParam: Param[CountVectorizer] =
    new Param(this, "countVectorizer", "Convert input tokens into weighted vectors.")
  setDefault(countVectorizerParam, new CountVectorizer())
  val ldaParam: Param[LDA] =
    new Param(this, "lda", "Convert each question into a weighted topic vector.")
  setDefault(ldaParam, new LDA())
  val logisticRegressionParam: Param[LogisticRegression] =
    new Param(this, "logisticRegression", "Combine question vectors pairs into a predicted probability.")
  setDefault(logisticRegressionParam, new LogisticRegression())
  private val questionsCols = Array("question1", "question2")

  def this() = this(Identifiable.randomUID("quoraquestionspairspipeline"))

  override def transformSchema(schema: StructType): StructType = assemblePipeline().transformSchema(schema)

  private val logger = org.log4s.getLogger
  override def fit(dataset: Dataset[_]): PipelineModel = {
    logger.info(s"Preparing to fit quora question pipeline with params:\n${explainParams()}")
    assemblePipeline().fit(dataset)
  }

  override def explainParams(): String = {
    Seq(
      $(cleanFeaturesTransformerParam),
      $(tokenizerParam),
      $(stopwordsRemoverParam),
      $(countVectorizerParam),
      $(ldaParam),
      $(logisticRegressionParam)).map(_.explainParams()).mkString("\n")
  }

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

  private def cleanerPipeline(): Array[PipelineStage] = {
    Array($(cleanFeaturesTransformerParam))
  }

  private def tokenizePipeline(): Array[PipelineStage] = {
    val mcTokenizer = new MultiColumnPipeline()
        .setStage($(tokenizerParam))
        .setInputCols(questions(""))
        .setOutputCols(questions("all_tokens"))
    val mcStopwordsRemover = new MultiColumnPipeline()
        .setStage($(stopwordsRemoverParam))
        .setInputCols(mcTokenizer.getOutputCols)
        .setOutputCols(questions("tokens"))
    Array(mcTokenizer, mcStopwordsRemover)
  }

  private def vectorizePipeline(): Array[PipelineStage] = {
    val mcCountVectorizer = new MultiColumnPipeline()
      .setInputCols(questions("tokens"))
      .setOutputCols(questions("vectors"))
      .setStage($(countVectorizerParam))
    Array(mcCountVectorizer)
  }

 private def ldaPipeline(): Array[PipelineStage] = {
   // The "em" optimizer supports serialization, is disk hungry and slow,
   // "online" is fast but cannot be serialized. We keep the latter as default, since this
   // model is only used to create a submission and nothing else.
   val optimizer = "online"
   val lda = $(ldaParam)
     .setOptimizer(optimizer)
     .setFeaturesCol("tmpinput").setTopicDistributionCol("tmpoutput")
   val mcLda = new MultiColumnPipeline()
     .setInputCols(questions("vectors"))
     .setOutputCols(questions("lda"))
     .setStage(lda, lda.getFeaturesCol, lda.getTopicDistributionCol)
   Array(mcLda)
  }

  private def questions(suffix: String) = questionsCols.map(_ + suffix)

  private def logisticRegressionPipeline(): Array[PipelineStage] = {
    val labelCol = "isDuplicateLabel"
    val assembler = new VectorAssembler().setInputCols(questions("lda")).setOutputCol("mergedlda")
    val labeler = new SQLTransformer().setStatement(
      s"SELECT *, cast(isDuplicate as int) $labelCol from __THIS__")
    val lr = new LogisticRegression()
        .setFeaturesCol("mergedlda").setProbabilityCol("p").setRawPredictionCol("raw")
        .setLabelCol(labelCol)
    Array(assembler, labeler, lr)
  }

  def copy(extra: ParamMap): QuoraQuestionsPairsPipeline = defaultCopy(extra)

  def setCleanFeaturesTransformer(value: CleanFeaturesTransformer): this.type = set(cleanFeaturesTransformerParam, value)

  def setTokenizer(value: Tokenizer): this.type = set(tokenizerParam, value)

  def setStopwordsRemover(value: StopWordsRemover): this.type = set(stopwordsRemoverParam, value)

  def setCountVectorizer(value: CountVectorizer): this.type = set(countVectorizerParam, value)

  def setLDA(value: LDA): this.type = set(ldaParam, value)

  def setLogisticRegression(value: LogisticRegression): this.type = set(logisticRegressionParam, value)
}
