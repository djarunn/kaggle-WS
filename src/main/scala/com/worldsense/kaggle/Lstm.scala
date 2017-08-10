package com.worldsense.kaggle

import com.intel.analytics.bigdl._
import com.intel.analytics.bigdl.dataset.{DataSet, FixedLength, Sample}
import com.intel.analytics.bigdl.nn._
import com.intel.analytics.bigdl.optim.Optimizer
import com.intel.analytics.bigdl.tensor.Tensor
import com.worldsense.kaggle.Lstm.LabeledSentence
import org.apache.spark.ml.param.{Param, ParamMap}
import org.apache.spark.ml.util.{DefaultParamsReadable, DefaultParamsWritable, Identifiable}
import org.apache.spark.ml.{DLEstimator, DLModel, Estimator}
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.Dataset
import org.apache.spark.sql.types.StructType

import scala.reflect.ClassTag

class Lstm(override val uid: String) extends Estimator[DLModel[Float]] with DefaultParamsWritable  {
  def this() = this(Identifiable.randomUID("lstm"))
  final val labelCol: Param[String] = new Param[String](this, "inputCol", "comma separate input column names")
  final val featuresCol: Param[String] = new Param[String](this, "featuresCol", "comma separate input column names")
  final val predictionCol: Param[String] = new Param[String](this, "predictionCol", "comma separate input column names")
  final val embeddingDim: Param[Int] = new Param[Int](this, "outputCol", "comma separate input column names")
  setDefault(embeddingDim, 300)
  final val hiddenDim: Param[Int] = new Param[Int](this, "hiddenDim", "comma separate input column names")
  setDefault(hiddenDim, 128)
  final val numClasses: Param[Int] = new Param[Int](this, "numClasses", "comma separate input column names")
  setDefault(numClasses, 2)
  final def getLabelCol: String = $(labelCol)
  final def getFeaturesCol: String = $(featuresCol)
  final def getPredictionCol: String = $(predictionCol)
  final def getEmbeddingDim: Int = $(embeddingDim)
  final def setLabelCol(col: String) = set(labelCol, col)
  final def setFeaturesCol(col: String) = set(featuresCol, col)
  final def setPredictionCol(col: String) = set(predictionCol, col)
  final def setEmbeddingDim(value: Int) = set(embeddingDim, value)
  final def setHiddenDim(value: Int) = set(hiddenDim, value)
  final def setNumClasses(value: Int) = set(numClasses, value)

  override def transformSchema(schema: StructType): StructType = {
    schema
    // assembleNeuralNetwork().transformSchema(schema)
  }
  override def fit(dataset: Dataset[_]): DLModel[Float] = {
    import dataset.sparkSession.implicits._
    val estimator = assembleNeuralNetwork(dataset.as[LabeledSentence])
    estimator.fit(dataset)
  }
  private def assembleNeuralNetwork(df: Dataset[LabeledSentence]): DLEstimator[Float] = {
    import df.sparkSession.implicits._
    val nn: Sequential[Float] = new Sequential[Float]()
      .add(Recurrent[Float]()
        .add(LSTM($(embeddingDim), $(hiddenDim))))
      .add(Select(2, -1))
      .add(Linear($(hiddenDim), 100))
      .add(Linear(100, $(numClasses)))
      // .add(Linear(4, $(numClasses)))
      .add(LogSoftMax())
    val batchSize = 1
    val criterion: Criterion[Float] = new ClassNLLCriterion[Float]()
    val sampleRdd: RDD[Sample[Float]] = df.rdd.map { ls =>
      val dataTensor = Tensor[Float](ls.vectors.flatten.toArray, Array(1, $(embeddingDim)))
      val labelTensor = Tensor[Float](ls.label.toArray, Array($(numClasses)))
      Sample(dataTensor, labelTensor)
    }
    sampleRdd.cache().count()
    val optimizer = Optimizer(nn, sampleRdd, criterion, batchSize)
    val model = optimizer.optimize()
    val estimator = new DLEstimator[Float](model, criterion, Array($(embeddingDim)), Array($(numClasses)))
    estimator.setFeaturesCol($(featuresCol))
    estimator.setLabelCol($(labelCol))
    estimator.setPredictionCol($(predictionCol))
    estimator.setBatchSize(batchSize)
    estimator
  }
  override def copy(extra: ParamMap): Lstm = defaultCopy(extra)
}
object Lstm extends DefaultParamsReadable[Lstm] {
  case class LabeledSentence(vectors: Seq[Seq[Float]], label: Seq[Float])
}
/*
object Lstm extends DefaultParamsReadable[Lstm] {
  class SeqDLEstimator[Float : ClassTag](
    model: Module[Float],
    criterion : Criterion[Float],
    featureSize : Array[Int],
    labelSize : Array[Int],
    override val uid: String = "DLEstimator")(implicit ev: TensorNumeric[Float])
      extends DLEstimator[Float](model, criterion, featureSize, labelSize, uid) {
    protected override def validateSchema(schema: StructType): Unit = {}
  }
}
*/
