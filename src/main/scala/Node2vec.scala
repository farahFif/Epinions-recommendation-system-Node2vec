import breeze.linalg.DenseMatrix
import org.apache.log4j.{Level, Logger}
import org.apache.spark.SparkContext
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.ml.evaluation.{BinaryClassificationEvaluator, MulticlassClassificationEvaluator}
import org.apache.spark.ml.feature._
import org.apache.spark.ml.tuning.{CrossValidator, ParamGridBuilder}
import org.apache.spark.rdd.RDD
import org.apache.spark.sql._
import org.apache.spark.sql.functions.udf
import org.apache.spark.sql.types.{IntegerType, StructField, StructType}
import breeze.linalg._
import breeze.numerics._
import org.apache.spark.mllib.random.UniformGenerator

object   Node2vec {

  def main(args: Array[String]) {

    Logger.getLogger("org.apache.spark").setLevel(Level.ERROR)

    val spark = SparkSession
      .builder()
      .appName("Application")
      .config("spark.master", "local")
      .getOrCreate()

    val data: RDD[(Int, Int)] = read_data("./train_epin.csv",spark)


    data.take(10).foreach(x => print(x))

    val total_nodes = 40334
    val emb_dim = 50
    val epoch = 20
    val learning_rate = 1.0
    val batch_size = 10000
    val neg_samples = 20

    val emb_in = create_embedding_matrix(emb_dim, total_nodes)
    val emb_out = create_embedding_matrix(emb_dim, total_nodes)

   // var emb_in_broadcast = sc.broadcast(emb_in)
   // var emb_out_broadcast = sc.broadcast(emb_out)
    val k: DenseMatrix[Double] = create_embedding_matrix(emb_dim,total_nodes)

    print("fffffffffffff", k(0,1))
    create_batchs(data, batch_size)
  }
  def read_data(path: String, spark: SparkSession): RDD[(Int, Int)] = {
    spark.read.format("csv")
      // the original data is store in CSV format
      // header: source_node, destination_node
      // here we read the data from CSV and export it as RDD[(Int, Int)],
      // i.e. as RDD of edges
      .option("header", "true")
      // State that the header is present in the file
      .schema(StructType(Array(
        StructField("source_node", IntegerType, false),
        StructField("destination_node", IntegerType, false)
      )))
      // Define schema of the input data
      .load(path)
      // Read the file as DataFrame
      .rdd.map(row => (row.getAs[Int](0), row.getAs[Int](1)))
    // Interpret DF as RDD
  }



  def create_batchs(data : RDD[(Int,Int)], batche_size : Int): Unit ={

    //val data_size = data.map(l => l.toString().length).collect()

    val index = data.zipWithIndex()
    val new_data = index.map(x => (x._2,x._1))

    val nb_batch = (new_data.count()/ batche_size).toInt

    val batchs = for (i <- 0 until nb_batch)
      yield new_data.filter(i == _._1 / nb_batch).map(x => (i,x._2))

    //batchs.foreach(rdd => for ((k,v) <- rdd.collect) printf("key: %s, value: %s\n", k, v))
  }



  def create_embedding_matrix(emb_dim : Int, total_nodes: Int):DenseMatrix[Double] ={
    val rand = new UniformGenerator()
    val vals = Array.fill(emb_dim * total_nodes)(rand.nextValue())
     val mat = new DenseMatrix(emb_dim,total_nodes, vals)
    return mat

  }

  def calculate_gradients(batch : RDD[(Int,(Int,Int))]): Unit ={


  }


  def calculate_deriv(  emb_in: DenseMatrix[Float], emb_out: DenseMatrix[Float]): Unit ={



  }
  def estimate_gradients_for_edge(
                                   source: Int,
                                   destination: Int,
                                   emb_in: DenseMatrix[Float],
                                   emb_out: DenseMatrix[Float]
                                 ) = {

    val in = emb_in(::, source)
    val out = emb_out(::, destination)

    val in_grad = out - out /(1 - sigmoid(in.t * out))
    val out_grad = in - in /(1- sigmoid(in.t * out))
    // return a tuple
    // Tuple((Int, DenseVector), (Int, DenseVector))
    // this tuple contains sparse gradients
    // in_grads is vector of gradients for
    // a node with id = source
    //((source, in_grads), (destination, out_grads))
    ((source, in_grad),(destination, out_grad))
  }

  def update_gradients(): Unit ={

  }
}
