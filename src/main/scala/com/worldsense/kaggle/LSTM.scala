// A wrapper for a spark ml estimator which fit it on the concatenation of multiple columns,
// and generates a model which transforms each of the columns with that result. For example,
// one can use it to train a tf/idf model on two text columns, and transform each of them using
// the idfs calculated over both of them.
package com.worldsense.kaggle

import org.apache.hadoop.fs.Path
import org.apache.spark.ml.feature.{ColumnPruner, SQLTransformer}
import org.apache.spark.ml.param.{Param, ParamMap}
import org.apache.spark.ml.util.{DefaultParamsReadable, DefaultParamsWritable, Identifiable, MLReadable, MLReader, MLWritable, MLWriter}
import org.apache.spark.ml.{Estimator, Pipeline, PipelineModel, PipelineStage}
import org.apache.spark.sql.Dataset
import org.apache.spark.sql.types.StructType

class LSTM(override val uid: String) extends Estimator[LSTMModel] with DefaultParamsWritable  {
  def this() = this(Identifiable.randomUID("lstm"))
  final val inputCol: Param[String] = new Param[String](this, "tmpInputCol", "comma separate input column names")
  final val outputCol: Param[String] = new Param[String](this, "tmpOutputCol", "comma separate input column names")
  final def getInputCol: String = $(tmpInputCol)
  final def getOutputCol: String = $(tmpOutputCol)
  final def setInputCol(col: String) = set(tmpInputCol, col)
  final def setOutputCol(col: String) = set(outputCols, col)
  override def transformSchema(schema: StructType): StructType = {
    val pipeline = makeWrappedPipeline(stage)
    pipeline.transformSchema(schema)
  }

  override def fit(dataset: Dataset[_]): LSTMModel = {
    // Fit on the concatenated input
    val model = fitInputCols(dataset)
    // Generate serial transformations to transform each input in its respective output.
    val pipeline = makeWrappedPipeline(model)
    // Not a real fit since all stages are transformers.
    pipeline.fit(dataset)
  }
  override def copy(extra: ParamMap): LSTM = {
    val instance = defaultCopy[LSTM](extra)
    instance.stage = stage.copy(extra)
    instance
  }
  override def write: MLWriter = new LSTMWriter(super.write, this)
}

object LSTM extends DefaultParamsReadable[LSTM]
