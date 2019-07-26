# coding=utf-8
# Copyright @akikaaa.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from bert.run_classifier import *
import tensorflow as tf

from tqdm import tqdm

class MyProcessor(DataProcessor):

    def get_test_examples(self, data_dir):
        return self.create_examples(
            self._read_tsv(os.path.join(data_dir, "test.csv")), "test")

    def get_train_examples(self, data_dir):
        """See base class."""
        return self.create_examples(
            self._read_tsv(os.path.join(data_dir, "train_small.csv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self.create_examples(
            self._read_tsv(os.path.join(data_dir, "dev.csv")), "dev")

    def get_pred_examples(self, data_dir):
        return self.create_examples(
            self._read_tsv(os.path.join(data_dir, "pred.csv")), "pred")

    def get_labels(self):
        """See base class."""
        label_array = ['999999', '201299', '900199', '300101', '950103', '100199',
                       '200299', '200501', '300203', '300102', '201999', '900102',
                       '200102', '201901', '400199', '200101', '201399', '100616',
                       '200409', '200801', '900103', '201006', '950199', '900101',
                       '950102', '100611', '100101', '300106', '200206', '100120',
                       '100105', '500801', '200702', '200203', '300104', '100699',
                       '100130', '300105', '200209', '201211', '201203', '201004',
                       '100612', '950101', '200403', '100599', '200499', '201206',
                       '200402', '900104', '100613', '100117', '200599', '201099',
                       '200202', '201002', '300103', '100201', '101207', '999901',
                       '950299', '100302', '200201', '900105', '100102', '100618',
                       '200213', '201218', '100502', '200502', '201001', '201216',
                       '100116', '201030', '100109', '100108', '200802', '200407',
                       '400101', '201602', '100799', '500806', '201005', '200904',
                       '300302', '700102', '200401', '201014', '201020', '500899',
                       '201212', '201213', '201219', '201208', '100104', '201003',
                       '201199', '100614', '900107', '300501', '100103', '100299',
                       '900108', '300199', '200799', '201207', '300301', '101202',
                       '950104', '200214', '201202', '201899', '900110', '400108',
                       '400102', '100608', '201016', '201209', '100106', '501003',
                       '201802', '100208', '300202', '700201', '100603', '500804',
                       '100602', '200220', '300599', '100999', '100399', '200212',
                       '201104', '201903', '100607', '600204', '201205', '201902',
                       '100606', '200703', '200899', '900111', '200207', '900106',
                       '201107', '102909', '201304', '400299', '100203', '202001',
                       '201703', '102899', '101299', '100905', '201007', '200507',
                       '202002', '100308', '200408', '201215', '101099', '200805',
                       '201008', '500802', '300201', '200215', '100202', '500803',
                       '200506', '102001', '950201', '500805', '200404', '101103',
                       '201204', '201799', '200804', '201501', '201015', '400201',
                       '201301', '900109', '201214', '201009', '200701', '200399',
                       '100704', '500808', '201302', '900112', '200210', '100301',
                       '200504', '201013', '100620', '100702', '501001', '200999',
                       '100499', '100609', '300401', '200208', '100901', '100110',
                       '400107', '501002', '101001', '300299', '400109', '100207',
                       '200808', '700101', '201701', '900113', '200420', '201010',
                       '200902', '201801', '102908', '100701', '200301', '300502',
                       '201905', '300499', '201012', '100111', '200508', '100405',
                       '102901', '400205', '200211', '200205', '200505', '950202',
                       '300399', '600699', '102819', '201011', '201102', '100119',
                       '100205', '300507', '300402', '100206', '300204', '600201',
                       '100118', '601301', '100605', '100210', '201210', '300205',
                       '200704', '300405', '100107', '100112', '100604', '100204',
                       '201101', '100903', '100899', '103002', '201217', '601302',
                       '400203', '600602', '601807', '600101', '601806', '201502',
                       '601305', '100113', '100619', '300506', '601306', '100904',
                       '400105', '100303', '600203', '202009', '100615', '100401',
                       '600904', '300503', '400103', '100306', '600701', '100115',
                       '600603', '100610', '601803', '200809', '700103', '600499',
                       '600406', '101002', '500807', '601799', '200410', '100402',
                       '101219', '100617', '100501', '300107', '100408', '600901',
                       '201105', '601303', '100902', '202102', '700202', '101101',
                       '601399', '600405', '101105', '101104', '400110', '100222',
                       '400104', '200901', '601801', '600299', '100209', '101208',
                       '101204', '100216', '100215', '601601', '100226', '300404',
                       '600601', '201601', '100223', '100404', '600604', '201303',
                       '100213', '100211', '600705', '100212', '600599', '102011',
                       '201599', '202109', '100407', '601701', '600999', '600199',
                       '400204', '601802', '600902', '200406', '601699', '202101',
                       '103004', '400206', '100224', '103099', '200503', '103001',
                       '102003', '101203', '606504', '700104', '100221', '999903',
                       '102016', '600301', '600399', '600501', '600803', '101206',
                       '600302', '600502', '201702', '606502', '601299', '600799',
                       '600202', '606599', '606510', '200303', '600103', '606501',
                       '601204', '201504', '600205', '102902', '601401', '601804',
                       '101902', '600899', '102999', '606508', '600802', '600801',
                       '400202', '601604', '102002', '600605', '600402', '200903',
                       '101201', '999902', '200405', '201904', '200803', '103003',
                       '103007', '200302', '103006', '601501', '103005', '601404',
                       '102611', '101899', '102820', '102601', '900123', '601304',
                       '600401', '100307', '600407']
        return label_array

    def create_examples(self, lines, set_type, file_base=True):
        """Creates examples for the training and dev sets. each line is label+\t+text_a+\t+text_b """
        examples = []
        for (i, line) in tqdm(enumerate(lines)):

            if file_base:
                if i == 0:
                    continue

            guid = "%s-%s" % (set_type, i)
            text = tokenization.convert_to_unicode(line[1])
            if set_type == "test" or set_type == "pred":
                label = "0"
            else:
                label = tokenization.convert_to_unicode(line[0])
            examples.append(
                InputExample(guid=guid, text_a=text, label=label))
        return examples


def main(_):
  tf.logging.set_verbosity(tf.logging.INFO)

  processors = {
      "setiment": MyProcessor
  }

  if not FLAGS.do_train and not FLAGS.do_eval and not FLAGS.do_predict:
    raise ValueError(
        "At least one of `do_train`, `do_eval` or `do_predict' must be True.")

  bert_config = modeling.BertConfig.from_json_file(FLAGS.bert_config_file)

  if FLAGS.max_seq_length > bert_config.max_position_embeddings:
    raise ValueError(
        "Cannot use sequence length %d because the BERT model "
        "was only trained up to sequence length %d" %
        (FLAGS.max_seq_length, bert_config.max_position_embeddings))

  tf.gfile.MakeDirs(FLAGS.output_dir)

  task_name = FLAGS.task_name.lower()

  if task_name not in processors:
    raise ValueError("Task not found: %s" % (task_name))

  processor = processors[task_name]()

  label_list = processor.get_labels()

  tokenizer = tokenization.FullTokenizer(
      vocab_file=FLAGS.vocab_file, do_lower_case=FLAGS.do_lower_case)

  tpu_cluster_resolver = None
  if FLAGS.use_tpu and FLAGS.tpu_name:
    tpu_cluster_resolver = tf.contrib.cluster_resolver.TPUClusterResolver(
        FLAGS.tpu_name, zone=FLAGS.tpu_zone, project=FLAGS.gcp_project)

  is_per_host = tf.contrib.tpu.InputPipelineConfig.PER_HOST_V2
  run_config = tf.contrib.tpu.RunConfig(
      cluster=tpu_cluster_resolver,
      master=FLAGS.master,
      model_dir=FLAGS.output_dir,
      save_checkpoints_steps=FLAGS.save_checkpoints_steps,
      tpu_config=tf.contrib.tpu.TPUConfig(
          iterations_per_loop=FLAGS.iterations_per_loop,
          num_shards=FLAGS.num_tpu_cores,
          per_host_input_for_training=is_per_host))

  train_examples = None
  num_train_steps = None
  num_warmup_steps = None
  if FLAGS.do_train:
    train_examples = processor.get_train_examples(FLAGS.data_dir)
    num_train_steps = int(
        len(train_examples) / FLAGS.train_batch_size * FLAGS.num_train_epochs)
    num_warmup_steps = int(num_train_steps * FLAGS.warmup_proportion)

  model_fn = model_fn_builder(
      bert_config=bert_config,
      num_labels=len(label_list),
      init_checkpoint=FLAGS.init_checkpoint,
      learning_rate=FLAGS.learning_rate,
      num_train_steps=num_train_steps,
      num_warmup_steps=num_warmup_steps,
      use_tpu=FLAGS.use_tpu,
      use_one_hot_embeddings=FLAGS.use_tpu)

  # If TPU is not available, this will fall back to normal Estimator on CPU
  # or GPU.
  estimator = tf.contrib.tpu.TPUEstimator(
      use_tpu=FLAGS.use_tpu,
      model_fn=model_fn,
      config=run_config,
      train_batch_size=FLAGS.train_batch_size,
      eval_batch_size=FLAGS.eval_batch_size,
      predict_batch_size=FLAGS.predict_batch_size)

  if FLAGS.do_train:
    train_file = os.path.join(FLAGS.output_dir, "train.tf_record")
    file_based_convert_examples_to_features(
        train_examples, label_list, FLAGS.max_seq_length, tokenizer, train_file)
    tf.logging.info("***** Running training *****")
    tf.logging.info("  Num examples = %d", len(train_examples))
    tf.logging.info("  Batch size = %d", FLAGS.train_batch_size)
    tf.logging.info("  Num steps = %d", num_train_steps)
    train_input_fn = file_based_input_fn_builder(
        input_file=train_file,
        seq_length=FLAGS.max_seq_length,
        is_training=True,
        drop_remainder=True)
    estimator.train(input_fn=train_input_fn, max_steps=num_train_steps)

  if FLAGS.do_eval:
    eval_examples = processor.get_dev_examples(FLAGS.data_dir)
    num_actual_eval_examples = len(eval_examples)

    eval_file = os.path.join(FLAGS.output_dir, "eval.tf_record")
    file_based_convert_examples_to_features(
        eval_examples, label_list, FLAGS.max_seq_length, tokenizer, eval_file)

    tf.logging.info("***** Running evaluation *****")
    tf.logging.info("  Num examples = %d (%d actual, %d padding)",
                    len(eval_examples), num_actual_eval_examples,
                    len(eval_examples) - num_actual_eval_examples)
    tf.logging.info("  Batch size = %d", FLAGS.eval_batch_size)

    # This tells the estimator to run through the entire set.
    eval_steps = None
    # However, if running eval on the TPU, you will need to specify the
    # number of steps.
    if FLAGS.use_tpu:
      assert len(eval_examples) % FLAGS.eval_batch_size == 0
      eval_steps = int(len(eval_examples) // FLAGS.eval_batch_size)

    eval_drop_remainder = True if FLAGS.use_tpu else False
    eval_input_fn = file_based_input_fn_builder(
        input_file=eval_file,
        seq_length=FLAGS.max_seq_length,
        is_training=False,
        drop_remainder=eval_drop_remainder)

    result = estimator.evaluate(input_fn=eval_input_fn, steps=eval_steps)

    output_eval_file = os.path.join(FLAGS.output_dir, "eval_results.txt")
    with tf.gfile.GFile(output_eval_file, "w") as writer:
      tf.logging.info("***** Eval results *****")
      for key in sorted(result.keys()):
        tf.logging.info("  %s = %s", key, str(result[key]))
        writer.write("%s = %s\n" % (key, str(result[key])))

  if FLAGS.do_predict:
    predict_examples = processor.get_test_examples(FLAGS.data_dir)
    num_actual_predict_examples = len(predict_examples)

    predict_file = os.path.join(FLAGS.output_dir, "predict.tf_record")
    file_based_convert_examples_to_features(predict_examples, label_list,
                                            FLAGS.max_seq_length, tokenizer,
                                            predict_file)

    tf.logging.info("***** Running prediction*****")
    tf.logging.info("  Num examples = %d (%d actual, %d padding)",
                    len(predict_examples), num_actual_predict_examples,
                    len(predict_examples) - num_actual_predict_examples)
    tf.logging.info("  Batch size = %d", FLAGS.predict_batch_size)

    predict_drop_remainder = True if FLAGS.use_tpu else False
    predict_input_fn = file_based_input_fn_builder(
        input_file=predict_file,
        seq_length=FLAGS.max_seq_length,
        is_training=False,
        drop_remainder=predict_drop_remainder)

    result = estimator.predict(input_fn=predict_input_fn)

    output_predict_file = os.path.join(FLAGS.output_dir, "test_results.tsv")
    with tf.gfile.GFile(output_predict_file, "w") as writer:
        tf.logging.info("***** Predict results *****")
        for prediction in result:
            output_line = "\t".join(str(class_probability) for class_probability in prediction) + "\n"
            writer.write(output_line)


if __name__ == "__main__":
  flags.mark_flag_as_required("data_dir")
  flags.mark_flag_as_required("task_name")
  flags.mark_flag_as_required("vocab_file")
  flags.mark_flag_as_required("bert_config_file")
  flags.mark_flag_as_required("output_dir")
  tf.app.run()

  '''
INFO:tensorflow:***** Eval results *****
INFO:tensorflow:  eval_accuracy = 0.8647313
INFO:tensorflow:  eval_loss = 0.7771957
INFO:tensorflow:  global_step = 6072
INFO:tensorflow:  loss = 0.77480406
  '''