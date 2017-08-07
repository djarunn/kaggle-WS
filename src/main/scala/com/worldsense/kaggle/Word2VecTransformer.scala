package com.worldsense.kaggle

import org.apache.spark.ml.Transformer
import org.apache.spark.ml.feature.Word2VecModel
import org.apache.spark.ml.param.{Param, ParamMap}
import org.apache.spark.ml.util.{DefaultParamsWritable, Identifiable}
import org.apache.spark.sql.types.StructType
import org.apache.spark.sql.{Dataset, SparkSession}

import scala.io.Source

class GloveEstimator(override val uid: String) extends Transformer with DefaultParamsWritable {
  private val gloveDir: Param[String] = new Param(this, "tokenizer", "estimator for selection")
  private val inputCol: Param[String] = new Param(this, "tokenizer", "estimator for selection")
  private val outputCol: Param[String] = new Param(this, "tokenizer", "estimator for selection")
  def this() = this(Identifiable.randomUID("word2vec"))

  def copy(extra: ParamMap): GloveEstimator = defaultCopy(extra)

  override def transformSchema(schema: StructType): StructType = schema
  override def fit(dataset: Dataset[_]): Word2VecModel = {
    new Word2VecModel()
  }
  override def load(): Map[String, Array[Float]] = {
    val glove = Word2VecTransformer.loadGloveVectors($(gloveDir))
    val gloveBC = ds.sparkSession.sparkContext.broadcast(glove)
    val dim = glove.headOption.map(_._2.length).getOrElse(0)
    val tokens = ds.as[List[String]]
    val vectors = tokens.map { token =>
       gloveBC.value.getOrElse(token, Array.fill[Float](dim)(0))
    }
    ds.withColumn("word2vec", vectors.col(vectors.columns.head))
  }
}

object Word2VecTransformer {
  def loadGloveVectors(spark: SparkSession, gloveDir: String): Dataset[String, Array[Float]] = {
    val filename = s"$gloveDir/glove.6B.100d.txt"
    spark.read.text(filename).map { row =>
      row.getAs()
    }
    val word2vec = Source.fromFile(filename, "ISO-8859-1").getLines.map { line =>
      val values = line.split(" ")
      val word = values(0)
      val coefs = values.slice(1, values.length).map(_.toFloat)
      (word, coefs)
    }
    word2vec.toMap
  }
}
