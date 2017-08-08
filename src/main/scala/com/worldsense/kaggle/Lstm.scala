package com.worldsense.kaggle

import com.intel.analytics.bigdl.Criterion
import com.intel.analytics.bigdl.nn._
import org.apache.spark.ml.param.{Param, ParamMap}
import org.apache.spark.ml.util.{DefaultParamsReadable, DefaultParamsWritable, Identifiable}
import org.apache.spark.ml.{DLEstimator, DLModel, Estimator}
import org.apache.spark.sql.Dataset
import org.apache.spark.sql.types.StructType

class Lstm(override val uid: String) extends Estimator[DLModel[Float]] with DefaultParamsWritable  {
  def this() = this(Identifiable.randomUID("lstm"))
  final val labelCol: Param[String] = new Param[String](this, "inputCol", "comma separate input column names")
  final val featuresCol: Param[String] = new Param[String](this, "outputCol", "comma separate input column names")
  final val predictionCol: Param[String] = new Param[String](this, "outputCol", "comma separate input column names")
  final val embeddingDim: Param[Int] = new Param[Int](this, "outputCol", "comma separate input column names")
  setDefault(embeddingDim, 300)
  final val hiddenDim: Param[Int] = new Param[Int](this, "hiddenDim", "comma separate input column names")
  setDefault(hiddenDim, 128)
  final val numClasses: Param[Int] = new Param[Int](this, "numClasses", "comma separate input column names")
  setDefault(hiddenDim, 2)
  final def getLabelCol: String = $(labelCol)
  final def getFeaturesCol: String = $(featuresCol)
  final def getPredictionCol: String = $(predictionCol)
  final def setLabelCol(col: String) = set(labelCol, col)
  final def setFeaturesCol(col: String) = set(featuresCol, col)
  final def setPredictionCol(col: String) = set(predictionCol, col)

  override def transformSchema(schema: StructType): StructType = assembleNeuralNetwork().transformSchema(schema)
  override def fit(dataset: Dataset[_]): DLModel[Float] = {
    val estimator = assembleNeuralNetwork()
    estimator.fit(dataset)
  }
  private def assembleNeuralNetwork(): DLEstimator[Float] = {
    val nn: Sequential[Float] = new Sequential[Float]()
      .add(Recurrent())
      .add(LSTM($(embeddingDim), $(hiddenDim)))
      .add(Select(2, -1))
      .add(Linear($(hiddenDim), 100))
      .add(Linear(100, $(numClasses)))
      .add(LogSoftMax())
    val criterion: Criterion[Float] = new ClassNLLCriterion[Float]()
    new DLEstimator[Float](nn, criterion, Array($(embeddingDim)), Array($(numClasses)))
      .setFeaturesCol($(featuresCol)).setLabelCol($(labelCol)).setPredictionCol($(predictionCol))
  }
  override def copy(extra: ParamMap): Lstm = defaultCopy(extra)
}
object Lstm extends DefaultParamsReadable[Lstm]
