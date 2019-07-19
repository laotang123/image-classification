from array import *
from collections import OrderedDict
import sys
import torch

# used for filtering unused weight tensors in multi-task setting
#   for example, filter_tensor_name = ["hidden2lm.weight", "hidden2lm.bias"]
filter_tensor_name = []
# used for specify the weight tensor of conv1d, because we can NOT figure it
#   out automatically, for example, conv1d_tensor_name = ["conv1.weight"]
conv1d_tensor_name = []
# if you implement the crf operator using two tensors: transition & alpha0, you
#   need to merge them into one tensor of shape: (#label + 1) * #label. and the
#   last row is corresponding to alpha0
crf_tensor_name = []
crf_tensor = []

is_lstm_op = {}

def get_op_name(tensor_name):
  dot = tensor_name.rfind(".")
  return tensor_name[:dot]

def get_suffix_name(tensor_name):
  dot = tensor_name.rfind(".")
  return tensor_name[dot+1:]

def is_bias_tensor(name_from_pytorch):
  if "bias" in name_from_pytorch:
    return True
  else:
    return False

def is_rnn_tensor(name_from_pytorch):
  if "h_l" in name_from_pytorch:
    return True
  else:
    return False

# precondition: it should be an rnn tensor
def is_lstm_tensor(tensor, op_name, suffix_name):
  global is_lstm_op
  if op_name in is_lstm_op:
    return is_lstm_op[op_name]
  dims = len(tensor.size())
  if 2 == dims and "hh" in suffix_name:
    axis1 = tensor.size()[0]
    axis2 = tensor.size()[1]
    res = True if axis1/axis2 == 4 else False
    is_lstm_op[op_name] = res
    return res
  elif op_name in is_lstm_op:
    return is_lstm_op[op_name]
  else:
    return False

def length_normalize(tensor_name, op_name):
  if len(tensor_name) < 32:
    return tensor_name.ljust(32, '\0'), op_name
  else:
    # more than 32, cut it to be 31 (1 for \0)
    need2cut = len(tensor_name) - 31
    res = op_name[:-need2cut] + tensor_name[len(op_name):]
    print "[WARNING] length >= 32, cut tensor_name & op_name!"
    return res.ljust(32, '\0'), op_name[:-need2cut]

def format_name(name, is_lstm):
  if not is_rnn_tensor(name):
    op_name = get_op_name(name)
    return length_normalize(name, op_name)
  start = name.index("h_l")
  end = name.find("_", start + 3)
  if -1 == end:
    tgt = name[start+1:]
    insert_pos = name.rfind(".")
    name = name[:insert_pos] + tgt + name[insert_pos:-len(tgt)]
  else:
    tgt = name[start+1:end]
    insert_pos = name.rfind(".")
    name = name[:insert_pos] + tgt + name[insert_pos:start+1] + name[end:]
  name = name.replace("weight", "w")
  name = name.replace("bias", "b")
  # logic for lstm bias tensor: remove "_ih"
  if is_lstm:
    dot = name.rfind('.')
    suffix = name[dot+1:]
    if "b_ih" == suffix[:4]:
      name = name[:dot+2] + suffix[4:]
  op_name = get_op_name(name)
  return length_normalize(name, op_name)



if len(sys.argv) != 3:
  print "usage: {} pytorch_model_file lnn_weight_file".format(sys.argv[0])
  sys.exit()

model_path = sys.argv[1]
res_path = sys.argv[2]
model = torch.load(model_path, map_location=lambda storage, loc : storage)
print type(model)

if isinstance(model, OrderedDict):
  dict = model
else:
  dict = model.state_dict()

tensor_name = []
tensor = []
is_rnn = []
is_lstm = []

bias_name = []
bias_tensor = []

ops = []

# for rnn, we can only figure out whether it is gru or lstm according to weight_hh,
# which comes after weight_ih. so, we scan the tensors to get info of weight_hh in
# the first time, and the others (weight_ih, bias_ih, bias_hh, etc) in the second time
for k, v in dict.items():
  if k in filter_tensor_name:
    continue
  if is_rnn_tensor(k):
    is_lstm_tensor(v, get_op_name(k), get_suffix_name(k))

print is_lstm_op

for k, v in dict.items():
  if k in filter_tensor_name:
    print "filter tensor: {}".format(k)
    continue
  print "orig_tensor_name: {}".format(k)
  if k in crf_tensor_name:
    crf_tensor.append(v)
    continue
  b_bias = is_bias_tensor(k)
  b_rnn = is_rnn_tensor(k)
  b_lstm = is_lstm_tensor(v, get_op_name(k), get_suffix_name(k))
  if b_bias and b_lstm:
    bias_name.append(k)
    bias_tensor.append(v)
  else:
    is_rnn.append(b_rnn)
    is_lstm.append(b_lstm)
    tensor_name.append(k)
    if k in conv1d_tensor_name and not b_bias:
      # transform out_channel|in_channel|kernel_size to
      # out_channel|kernel_size|in_channel for conv1d weight
      tensor.append(torch.transpose(v, 1, 2))
    else:
      tensor.append(v)
print
# merge bias_ih & bias_hh for lstm
for i in range(0, len(bias_name), 2):
  tensor_name.append(bias_name[i])
  tensor.append(bias_tensor[i] + bias_tensor[i+1])
  is_rnn.append(True)
  is_lstm.append(True)
if len(crf_tensor) > 1:
  # merge transition & alpha0 for crf
  tensor_name.append("crf.weight")
  label_size = crf_tensor[0].size()[0]
  tensor.append(torch.cat((crf_tensor[0], crf_tensor[1].view(-1, label_size)), 0))
  is_rnn.append(False)
  is_lstm.append(False)

print "#tensor: {}".format(len(tensor_name))

f = file(res_path, 'wb')

# write magic & tensor number
magic_arr = array('H')
magic_arr.append(0x1)
magic_arr.append(len(tensor_name))
magic_arr.tofile(f)

# write tensor name & size & value
name_arr = array('c')
size_arr = array('I')
val_arr = array('f')
for i in range(len(tensor)):
  if "feature_embed.weights" == tensor_name[i]:
    tensor_name[i] = "feature_embed.weight"
  op_name = get_op_name(tensor_name[i])
  print "tensor_{} (of operator [{}]): \torig_name: {}\tshape: {}".format(i, op_name, tensor_name[i], tensor[i].size())
  name, op_name = format_name(tensor_name[i], is_lstm[i])
  if not op_name in ops:
    ops.append(op_name)
  for j in range(len(name)):
    name_arr.append(name[j])
  tensor_i = tensor[i].numpy().flatten().tolist()
  print "\t(of operator [{}])\tfinal_name: {}\tsize: {}\tis_rnn: {}\tis_lstm: {}\n".format(op_name, name, len(tensor_i), is_rnn[i], is_lstm[i])
  size_arr.append(len(tensor_i))
  # transform i|f|g|o to i|f|o|g for lstm
  if True == is_lstm[i]:
    part = len(tensor_i) / 4
    val_arr.fromlist(tensor_i[:2*part])
    val_arr.fromlist(tensor_i[3*part:])
    val_arr.fromlist(tensor_i[2*part:3*part])
  else:
    val_arr.fromlist(tensor_i)
#  for j in range(len(tensor_i)):
#    if j != len(tensor_i) - 1:
#      print "%.6f" % tensor_i[j],
#    else:
#      print "%.6f" % tensor_i[j]
#print

print "#operator: {}".format(len(ops))
for i in range(len(ops)):
  print ops[i]

name_arr.tofile(f)
size_arr.tofile(f)
val_arr.tofile(f)
f.close()
