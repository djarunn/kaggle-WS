package com.worldsense.kaggle

import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.ml.clustering.LDA
import org.apache.spark.ml.feature._
import org.apache.spark.ml.util.{MLReadable, MLReader}
import org.apache.spark.ml.{Pipeline, PipelineStage}


class QuoraQuestionsPairsPipeline(override val uid: String) extends Pipeline  {
  setStages(assemblePipeline())
  // Expose the underlying models so parameters can be tuned. Reflection is needed
  // due to the serialization mechanism involved.
  def tokenizer = getStages(1).asInstanceOf[MultiColumnPipeline].asInstanceOf[Tokenizer]
  def countVectorizer = getStages(2).asInstanceOf[MultiColumnPipeline].getStage.asInstanceOf[CountVectorizer]
  def latentDirichletAllocator = getStages(3).asInstanceOf[MultiColumnPipeline].getStage.asInstanceOf[LDA]
  def logisticRegression = getStages(4).asInstanceOf[LogisticRegression]

  private def assemblePipeline(): Array[PipelineStage] = {
    val questions = Array("question1", "question2")
    val tokenized = questions.map(q => s"${q}_stopworded_tokens")
    val vectorized = tokenized.map(q => s"${q}_vector")
    val ldaed = vectorized.map(q => s"${q}_lda")

    val tokenizer = tokenizePipeline(questions)
    val vectorizer = vectorizePipeline(tokenized)
    val lda = ldaPipeline(vectorized)
    val lr = probabilityPipeline(ldaed)
    Seq(tokenizer, vectorizer, lda, lr).flatten.toArray
  }
  private val logger = org.log4s.getLogger
  def tokenizePipeline(columns: Array[String]): Array[PipelineStage] = {
    val mcTokenizer = new MultiColumnPipeline()
        .setStage(new Tokenizer())
        .setInputCols(columns).setOutputCols(columns.map(_ + "_tokens"))
    val mcStopwordsRemover = new MultiColumnPipeline()
        .setStage(new StopWordsRemover())
        .setInputCols(mcTokenizer.getOutputCols).setOutputCols(columns.map(_ + "_stopworded_tokens"))
    Array(tokenizer, mcStopwordsRemover)
  }

  def vectorizePipeline(columns: Array[String]): Array[PipelineStage] = {
    val mcCountVectorizer = new MultiColumnPipeline()
      .setInputCols(columns).setOutputCols(columns.map(_ + "_vector"))
      .setStage(countVectorizer)
    Array(mcCountVectorizer)
  }

 def ldaPipeline(columns: Array[String]): Array[PipelineStage] = {
   // The "em" optimizer supports serialization, is disk hungry and slow,
   // "online" is fast but cannot be serialized. We keep the latter as default, since this
   // model is only used to create a submission and nothing else.
   val optimizer = "online"
   val lda = new LDA()
     .setOptimizer(optimizer)
     .setFeaturesCol("tmpinput").setTopicDistributionCol("tmpoutput")
   val mcLda = new MultiColumnPipeline()
     .setInputCols(columns).setOutputCols(columns.map(_ + "_lda"))
     .setStage(lda, lda.getFeaturesCol, lda.getTopicDistributionCol))
  }

  def probabilityPipeline(columns: Array[String]): Array[PipelineStage] = {
    val labelCol = "isDuplicateLabel"
    val assembler = new VectorAssembler().setInputCols(columns).setOutputCol("mergedlda")
    val labeler = new SQLTransformer().setStatement(
      s"SELECT *, cast(isDuplicate as int) $labelCol from __THIS__")
    val lr = new LogisticRegression()
        .setFeaturesCol("mergedlda").setProbabilityCol("p").setRawPredictionCol("raw")
        .setLabelCol(labelCol)
    Array(assembler, labeler, lr)
  }
}

object QuoraQuestionsPairsPipeline extends MLReadable[QuoraQuestionsPairsPipeline] {
  override def read: MLReader[QuoraQuestionsPairsPipeline] = new QuoraQuestionsPairsPipelineReader
  class QuoraQuestionsPairsPipelineReader extends MLReader[QuoraQuestionsPairsPipeline] {
    override def load(path: String): QuoraQuestionsPairsPipeline = {
      val pipeline = Pipeline.load(path)
      new QuoraQuestionsPairsPipeline().setStages(pipeline.stages)
    }
  }
}
