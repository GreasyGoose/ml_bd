package linear_regression

import breeze.linalg.Matrix.castOps
import breeze.linalg.{DenseVector, _}
import breeze.numerics.{pow, exp, log}
import breeze.stats.mean

import java.io._

object Main {



  def read_data(filename: String): DenseMatrix[Double] = {
    var df = csvread(new File(filename),',', skipLines=1)
    df = min_max_scaler(df)
    df
  }

  def min_max_scaler(df: DenseMatrix[Double]) : DenseMatrix[Double] = {
    // X_std = (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0))
    // X_scaled = X_std * (new_max - new_min) + new_min
    val selected_columns = List(0, 2) //for age and bmi
    for(col_idx<-selected_columns)
    {
      val col = df(::, col_idx.toInt)
      val x_min = col.min
      val x_max = col.max
      val x_std = (col - x_min) / (x_max - x_min)
      val x_scaled = x_std // when scaling from 0 to 1
      df(::, col_idx.toInt) := x_scaled
    }
    df

  }

  def train(df: DenseMatrix[Double]): Unit = {
    val (x_train, y_train, x_val, y_val) = train_test_split(df)
    val (weights,bias) = fit(x_train,y_train, x_val, y_val)
    save_model(weights,bias)
  }

  def apply(df: DenseMatrix[Double]): Unit = {
    val (w, b) = load_model()
    val x_test = df(::, 0 until df.cols - 1)
    val y_pred = predict(x_test, w, b)
    val filename = "output/preds.csv"
    csvwrite(new File(filename), y_pred.asDenseMatrix, separator = ',')

  }

  def train_test_split(df: DenseMatrix[Double], val_size: Double = 0.2, shuffle: Boolean = false):
  (DenseMatrix[Double], DenseVector[Double], DenseMatrix[Double], DenseVector[Double]) = {
    var indexList = DenseVector.range(0, df.rows,1)
    if (shuffle) {
      indexList = breeze.linalg.shuffle(indexList)
    }

    val cutoff = (val_size * df.rows).toInt
    val val_indexes = indexList(0 until cutoff)
    val train_indexes = indexList(cutoff until df.rows)

    var df_train = DenseMatrix.zeros[Double](df.rows - cutoff, df.cols)
    var df_val = DenseMatrix.zeros[Double](cutoff, df.cols)

    var count = 0 //Int
    for(row_idx <- train_indexes )
    {
      df_train(count, ::) := df(row_idx.toInt, ::)
      count += 1
    }

    count = 0
    for(row_idx <- val_indexes )
    {
      df_val(count, ::) := df(row_idx.toInt, ::)
      count += 1
    }
    (df_train(::, 0 until df.cols - 1), df_train(::,-1),
      df_val(::, 0 until df.cols - 1), df_val(::,-1))
  }


  def predict(x: DenseMatrix[Double], w: DenseVector[Double], b: Double): DenseVector[Double] = {
    val y_pred: DenseVector[Double] = x * w + b
    y_pred
  }

  def update_weights(x: DenseMatrix[Double], y: DenseVector[Double], w: DenseVector[Double], b: Double,
                     learning_rate: Double): (DenseVector[Double], Double) = {

    val y_pred = predict(x, w, b)

    val dw: DenseVector[Double] = -(2.0 * x.t * (y - y_pred)) / (x.rows).toDouble
    val db: Double = -2.0 * sum(y - y_pred) / (x.rows).toDouble

    val new_w = w - learning_rate * dw
    val new_b = b - learning_rate * db

    (new_w, new_b)
  }

  def fit(x: DenseMatrix[Double], y: DenseVector[Double], x_val: DenseMatrix[Double], y_val: DenseVector[Double]): (DenseVector[Double], Double) = {
    var w: DenseVector[Double] = DenseVector.zeros[Double](x.cols)
    var b: Double = 0.0
    val learning_rate: Double = 0.1
    val num_of_iterations = 1000

    for (i <- 1 to num_of_iterations) {
      val (new_w, new_b) = update_weights(x, y, w, b, learning_rate)
      w = new_w
      b = new_b
      if (i % 100 == 0) {

        val score = validate(x_val, y_val, w, b)
        val message = f"Epoch $i%s: R2 score $score%2.6f"
        val filename = f"output/epoch_$i%s.txt"
        val file = new File(filename)
        val bw = new BufferedWriter(new FileWriter(file))
        bw.write(message)
        bw.close()
        println(message)
//        csvwrite(new File(filename), i, separator = ' ')
      }
    }
    (w, b)
  }

  def validate(x: DenseMatrix[Double], y: DenseVector[Double], w: DenseVector[Double], b: Double): Double = {
    val y_pred = predict(x, w, b)
    val score = get_score(y, y_pred)
    score
  }

  def get_score(y: DenseVector[Double], y_pred: DenseVector[Double]): Double = {
    //  calc R2 score
    val a = y - y_pred
    val b = y - mean(y)
    1.0 - sum(pow(a, 2)) / sum(pow(b, 2))
  }

  def save_model(w: DenseVector[Double], b: Double): Unit = {
//    val dir: File = "model"
//      .toFile
//      .createIfNotExists(true)
    val filename1 = "model/weights.csv"
    val filename2 = "model/bias.csv"
    csvwrite(new File(filename1), w.asDenseMatrix, separator = ',')
    csvwrite(new File(filename2), DenseMatrix(b), separator = ',')
  }

  def load_model(): (DenseVector[Double], Double) = {
    val filename1 = "model/weights.csv"
    val filename2 = "model/bias.csv"
    var w = csvread(new File(filename1),',')
    var b = csvread(new File(filename2),',')
    (w.toDenseVector, b.valueAt(0))
  }

  val usage = """
      Usage:
      train path_to_train_data.csv
      apply path_to_data.csv
    """

  def main(args: Array[String]) {
    if (args.length == 0) println(usage)
    for(arg<-args)
    {
      println(arg);
    }

    val df = read_data(args(1).toString())

    if (args(0) == "train") {
      train(df)
    }
    else if (args(0) == "apply") {
      apply(df)
    }
    else {
      println("Unknown command", args(0));
    }

  }


}
