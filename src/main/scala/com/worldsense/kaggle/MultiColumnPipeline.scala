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

class MultiColumnPipeline(override val uid: String) extends Estimator[PipelineModel] with DefaultParamsWritable  {
  import MultiColumnPipeline.{EstimatorWithInputOutput, MultiColumnPipelineWriter}
  def this() = this(Identifiable.randomUID("multicolumnpipeline"))
  final val tmpInputCol: Param[String] = new Param[String](this, "tmpInputCol", "comma separate input column names")
  final val tmpOutputCol: Param[String] = new Param[String](this, "tmpOutputCol", "comma separate input column names")
  final def getTmpInputCol: String = $(tmpInputCol)
  final def getTmpOutputCol: String = $(tmpOutputCol)
  final def setTmpInputCol(col: String) = set(tmpInputCol, col)
  final def setTmpOutputCol(col: String) = set(outputCols, col)
  final val inputCols: Param[String] = new Param[String](this, "inputCols", "comma separate input column names")
  final val outputCols: Param[String] = new Param[String](this, "outputCols", "comma separate output column names")
  final def getInputCols: Seq[String] = $(inputCols).split(',')
  final def getOutputCols: Seq[String] = $(outputCols).split(',')
  final def setInputCols(cols: Seq[String]) = set(inputCols, cols.mkString(","))
  final def setOutputCols(cols: Seq[String]) = set(outputCols, cols.mkString(","))
  // The parameter holding the estimator that will be applied to the input columns. Does not use
  // the Param machinery since that works only with basic types.
  private var stage: PipelineStage = new Pipeline()   // identity transformer
  def getStage: PipelineStage = stage
  // Sets the stage, assuming it is already configured to read from tmpInput and write into tmpOutput.
  def setStage(stage: PipelineStage, tmpInput: String, tmpOutput: String): this.type = {
    this.stage = stage
    set(tmpInputCol, tmpInput)
    set(tmpOutputCol, tmpOutput)
    this
  }
  // Convenience version of setStage which assumes that the estimator implements
  // HasInputCol and HasOutputCol. Since those are private traits, we must resort to structural
  // typing (aka scala duck typing).
  def setStage[T <: PipelineStage with EstimatorWithInputOutput[T]](stage: T): this.type = {
    stage.setInputCol(s"tmpinput4${stage.uid}")
    stage.setOutputCol(s"tmpoutput4${stage.uid}")
    setStage(stage, stage.getInputCol, stage.getOutputCol)
    this
  }
  override def transformSchema(schema: StructType): StructType = {
    val pipeline = makeWrappedPipeline(stage)
    pipeline.transformSchema(schema)
  }

  // Fit the estimator on the concatenated input cols
  final protected def fitInputCols[T](dataset: Dataset[_]): PipelineModel = {
    // Use pipeline api so we can fit a model of generic type, including even transformers.
    val pipeline = new Pipeline().setStages(Array(stage))
    // Create a new dataset for each input column, copying its data to its sole tmp input column.
    val trainDS = getInputCols.map(c => dataset.select(c).withColumnRenamed(c, getTmpInputCol))
    // Concatenate the copies of all input columns and fit on it.
    pipeline.fit(trainDS.reduce(_.union(_)))
  }
  // Fit the concatenation of input columns on the dataset in a model that transforms each of them
  // in its respective output column.
  override def fit(dataset: Dataset[_]): PipelineModel = {
    // Fit on the concatenated input
    val model = fitInputCols(dataset)
    // Generate serial transformations to transform each input in its respective output.
    val pipeline = makeWrappedPipeline(model)
    // Not a real fit since all stages are transformers.
    pipeline.fit(dataset)
  }
  private def makeWrappedPipeline(wrapped: PipelineStage): Pipeline = {
    val stages = getInputCols.indices.map { i =>
      // Wrap a copy of the pipeline stage with column renames so it transforms
      // inputCol(i) into outputCol(i), just temporarily using tmpInputCol and tmpOutputCol.
      wrapWithColumnRename(
        wrapped.copy(ParamMap.empty),
        getTmpInputCol,  getTmpOutputCol,
        getInputCols(i), getOutputCols(i)
      )
    }
    new Pipeline().setStages(stages.flatten.toArray)
  }
  private def wrapWithColumnRename(
      stage: PipelineStage,
      inputCol: String, outputCol: String,
      renamedInputCol: String, renamedOutputCol: String): Seq[PipelineStage] = {
    Seq(
      new SQLTransformer().setStatement(s"SELECT *, $renamedInputCol as $inputCol FROM __THIS__"),
      stage,
      new SQLTransformer().setStatement(s"SELECT *, $outputCol as $renamedOutputCol FROM __THIS__"),
      new ColumnPruner().setInputCols(Array(inputCol, outputCol))
    )
  }

  override def copy(extra: ParamMap): MultiColumnPipeline = {
    val instance = defaultCopy[MultiColumnPipeline](extra)
    instance.stage = stage.copy(extra)
    instance
  }
  override def write: MLWriter = new MultiColumnPipelineWriter(super.write, this)
}

object MultiColumnPipeline extends DefaultParamsReadable[MultiColumnPipeline] {
  // Mimic layout of private traits HasInputCol and HasOutputCol for usage in type bounds.
  private[kaggle] type EstimatorWithInputOutput[T] = {
    def getInputCol: String
    def setInputCol(col: String): T
    def getOutputCol: String
    def setOutputCol(col: String): T
  }
  private class MultiColumnPipelineWriter(writer: MLWriter, instance: MultiColumnPipeline) extends MLWriter {
    override protected def saveImpl(path: String): Unit = {
      writer.save(path)
      // Uses the pipeline class for serialization, since most of the spark ml serialization
      // machinery is private.
      val pipeline = new Pipeline().setStages(Array(instance.stage))
      pipeline.write.save(new Path(path, "multicolumn").toString)
    }
  }
  override def read: MLReader[MultiColumnPipeline] = new MultiColumnPipelineReader(super.read)
  private[MultiColumnPipeline] class MultiColumnPipelineReader(val reader: MLReader[MultiColumnPipeline]) extends MLReader[MultiColumnPipeline] {
    override def load(path: String): MultiColumnPipeline = {
      val instance = reader.load(path)
      // Read estimator and parameters using the Pipeline class and uses them to create a new
      // MultiColumnPipeline instance.
      val pipeline: Pipeline = Pipeline.read.load(new Path(path, "multicolumn").toString)
      instance.stage = pipeline.getStages.head
      instance
    }
  }
}
