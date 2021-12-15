package org.apache.spark.ml.made

import breeze.linalg.sum
import breeze.linalg.{functions}
import org.apache.spark.ml.attribute.AttributeGroup
import org.apache.spark.ml.linalg.{DenseVector, Vector, VectorUDT, Vectors}
import org.apache.spark.ml.param.{BooleanParam, DoubleParam, Param, ParamMap, IntParam}
import org.apache.spark.ml.param.shared.{HasInputCol, HasOutputCol}
import org.apache.spark.ml.stat.Summarizer
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.util._
import org.apache.spark.ml.{Estimator, Model}
import org.apache.spark.mllib
import org.apache.spark.mllib.stat.MultivariateOnlineSummarizer
import org.apache.spark.sql.catalyst.encoders.ExpressionEncoder
import org.apache.spark.sql.types.StructType
import org.apache.spark.sql.{DataFrame, Dataset, Encoder, Row}

trait LinearRegressionParams extends HasInputCol with HasOutputCol {
  def setInputCol(value: String) : this.type = set(inputCol, value)
  def setOutputCol(value: String): this.type = set(outputCol, value)

  val predCol = new Param[String](
    this, "predCol","Whenever to predict")

  // Predictions
  def setPredCol(value: String): this.type = set(predCol, value)
  setDefault(predCol -> "")

  // Learning rate value
  val lr = new DoubleParam(this, "lr", "learning rate value")
  def setLr(value: Double) : this.type = set(lr, value)
  setDefault(lr -> 1e-1)
  

  val num_of_iterations = new IntParam(this, "num_of_iterations", "number of iterations before stop")
  def setIter(value: Int) : this.type = set(num_of_iterations, value)
  setDefault(num_of_iterations -> 100)

  protected def validateAndTransformSchema(schema: StructType): StructType = {
    SchemaUtils.checkColumnType(schema, getInputCol, new VectorUDT())
    SchemaUtils.appendColumn(schema, schema(getInputCol).copy(name = getOutputCol))
    schema

//    if (schema.fieldNames.contains($(outputCol))) {
//      SchemaUtils.checkColumnType(schema, getOutputCol, new VectorUDT())
//      schema
//    } else {
//      SchemaUtils.appendColumn(schema, schema(getInputCol).copy(name = getOutputCol))
//    }
  }
}

class LinearRegression(override val uid: String) extends Estimator[LinearRegressionModel] with LinearRegressionParams
  with DefaultParamsWritable {

  def this() = this(Identifiable.randomUID("LinearRegression"))

  override def fit(dataset: Dataset[_]): LinearRegressionModel = {

    // Used to convert untyped dataframes to datasets with vectors
    implicit val encoder : Encoder[Vector] = ExpressionEncoder()

    val new_col = "res"
    val assembler = new VectorAssembler()
      .setInputCols(Array($(inputCol), $(outputCol)))
      .setOutputCol(new_col)

    //input_col - features, preds


    val transformed = assembler.
      transform(dataset)
    val vectors: Dataset[Vector] = transformed.select(new_col).as[Vector]

    val dim: Int = AttributeGroup.fromStructField((dataset.schema($(inputCol)))).numAttributes.getOrElse(
      vectors.first().size - 1
//        vectors.first().size
    )

    var coeff = breeze.linalg.DenseVector.rand[Double](dim + 1)

    for (i <- 1 to $(num_of_iterations)) {
      val res = vectors.rdd.mapPartitions((data: Iterator[Vector]) => {
        val summarizer = new MultivariateOnlineSummarizer()
        data.foreach(vec => {
          val x = vec.asBreeze(0 to coeff.size - 1).toDenseVector
          val y = vec.asBreeze(-1)
          val diff = x * (sum(x * coeff) - y)
          summarizer.add(mllib.linalg.Vectors.fromBreeze(diff))
        })
        Iterator(summarizer)
      }).reduce(_ merge _)

      coeff = coeff - res.mean.asBreeze * $(lr)
    }


    copyValues(new LinearRegressionModel(coeff).setParent(this))
    //    val Row(row: Row) =  dataset
    //      .select(Summarizer.metrics("mean", "std").summary(dataset($(inputCol))))
    //      .first()
    //
    //    copyValues(new LinearRegressionModel(row.getAs[Vector](0).toDense, row.getAs[Vector](1).toDense)).setParent(this)
  }

  override def copy(extra: ParamMap): Estimator[LinearRegressionModel] = defaultCopy(extra)

  override def transformSchema(schema: StructType): StructType = validateAndTransformSchema(schema)

}

object LinearRegression extends DefaultParamsReadable[LinearRegression]

class LinearRegressionModel private[made](
                                         override val uid: String,
                                         val coeff: breeze.linalg.DenseVector[Double])
  extends Model[LinearRegressionModel] with LinearRegressionParams with MLWritable {


  private[made] def this(coeff: breeze.linalg.DenseVector[Double]) =
    this(Identifiable.randomUID("LinearRegressionModel"),coeff)

  override def copy(extra: ParamMap): LinearRegressionModel = copyValues(
    new LinearRegressionModel(coeff), extra)

  override def transform(dataset: Dataset[_]): DataFrame = {

    val w = coeff(0 to coeff.size - 2) //ie w
    val b = coeff(-1) //ie b

    val transformUdf = {
      dataset.sqlContext.udf.register(uid + "_transform",
        (x : Vector) => {
          sum(x.asBreeze.toDenseVector * w) + b
        }
      )
    }
    dataset.withColumn($(predCol), transformUdf(dataset($(inputCol))))
  }

  override def transformSchema(schema: StructType): StructType = validateAndTransformSchema(schema)

  override def write: MLWriter = new DefaultParamsWriter(this) {
    override protected def saveImpl(path: String): Unit = {
      super.saveImpl(path)

      val vectors = Tuple1(Vectors.fromBreeze(coeff))

      sqlContext.createDataFrame(Seq(vectors)).write.parquet(path + "/vectors")

    }
    }

}

object LinearRegressionModel extends MLReadable[LinearRegressionModel] {
  override def read: MLReader[LinearRegressionModel] = new MLReader[LinearRegressionModel] {
    override def load(path: String): LinearRegressionModel = {
      val metadata = DefaultParamsReader.loadMetadata(path, sc)

      val vectors = sqlContext.read.parquet(path + "/vectors")

      // Used to convert untyped dataframes to datasets with vectors
      implicit val encoder : Encoder[Vector] = ExpressionEncoder()

      val coeff =  vectors.select(vectors("_1").as[Vector])

      val model = new LinearRegressionModel(coeff.first().asBreeze.toDenseVector)
      metadata.getAndSetParams(model)
      model
    }
  }
}