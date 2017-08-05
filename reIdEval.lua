-- Copyright (c) 2017 Shuangjie Xu, EIC, Huazhong University of Science and Technology, China
-- Contact: shuangjiexu@hust.edu.cn
-- If you use this code please cite:
-- "Jointly Attentive Spatial-Temporal Pooling Networks for Video-based Person Re-Identification",
-- Shuangjie Xu, Yu Cheng, Kang Gu, Yang Yang, Shiyu Chang and Pan Zhou,
-- 2017 IEEE International Conference on Computer Vision (ICCV)
--
-- Copyright (c) 2016 Niall McLaughlin, CSIT, Queen's University Belfast, UK
-- Contact: nmclaughlin02@qub.ac.uk
-- If you use this code please cite:
-- "Recurrent Convolutional Network for Video-based Person Re-Identification",
-- N McLaughlin, J Martinez Del Rincon, P Miller, 
-- IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2016
-- 
-- This software is licensed for research and non-commercial use only.
-- 
-- The above copyright notice and this permission notice shall be included in
-- all copies or substantial portions of the Software.
-- 
-- THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
-- IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
-- FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
-- AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
-- LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
-- OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
-- THE SOFTWARE.

require 'torch'
require 'nn'
require 'nnx'
require 'optim'
require 'cunn'
require 'cutorch'
require 'image'
require 'paths'
require 'rnn'
require 'inn'

require 'layers/MetrixMultiply'
require 'train'
require 'test'

local datasetUtils = require 'datasets/datasetUtils'
local prepDataset = require 'datasets/prepareDataset'

-- set the GPU
cutorch.setDevice(1)

cmd = torch.CmdLine()
cmd:option('-dataset',1,'1 -  ilids, 2 - prid, 3 - mars')
cmd:option('-dirRGB','','dir path to the sequences of original datasets')
cmd:option('-dirOF','','dir path to the sequences of optical flow')
cmd:option('-sampleSeqLength',16,'length of sequence to train network')
cmd:option('-weight','weights/convNet_basicnet.dat','name to save dataset file')
cmd:option('-usePredefinedSplit',false,'Use predefined test/training split loaded from a file')
cmd:option('-disableOpticalFlow',false,'use optical flow features or not')
cmd:option('-seed',1,'random seed')

opt = cmd:parse(arg)
print(opt)

function isnan(z)
    return z ~= z
end

torch.manualSeed(opt.seed)
cutorch.manualSeed(opt.seed)

-- change these paths to point to the place where you store i-lids or prid datasets
homeDir = paths.home
filePrefix='.png'

if opt.dataset == 1 then
    seqRootRGB = 'data/i-LIDS-VID/sequences/'
    seqRootOF = 'data/i-LIDS-VID-OF-HVP/sequences/'
elseif opt.dataset == 2 then
    seqRootRGB = homeDir .. 'data/PRID2011/multi_shot/'
    seqRootOF = homeDir .. 'data/PRID2011-OF-HVP/multi_shot/'
elseif opt.dataset == 3 then
    seqRootRGB = homeDir .. 'data/MARS/sequences/'
    seqRootOF = homeDir .. 'data/MARS-OF-HVP/sequences/'
    filePrefix='.jpg'
else
    print('Unknown datasets')
    os.exit(0)
end

if opt.seqRootRGB and opt.seqRootOF then
    seqRootRGB = opt.seqRootRGB
    seqRootOF = opt.seqRootOF
end


print('loading Dataset - ',seqRootRGB,seqRootOF)
dataset = prepDataset.prepareDataset(seqRootRGB,seqRootOF,filePrefix)
print('dataset loaded',#dataset,seqRootRGB,seqRootOF)

if opt.usePredefinedSplit then
    -- useful for debugging to run with exactly the same test/train split
    print('loading predefined test/training split')
    local datasetSplit
    if opt.dataset == 1 then
        datasetSplit = torch.load('./data/dataSplit.th7')
    else
        datasetSplit = torch.load('./data/dataSplit_PRID2011.th7')
    end    
    testInds = datasetSplit.testInds
    trainInds = datasetSplit.trainInds
else
    print('randomizing test/training split')
    trainInds,testInds = datasetUtils.partitionDataset(#dataset,0.5)
end

-- Load the Model and Convnet (which is part of the model) from a file
--trainedModel = torch.load(saveFileNameModel)
trainedConvnet = torch.load(opt.weight)
--trainedBaseNet = torch.load(saveFileNameBasenet)

------------------------------------------------------------------------------------------------------------------------------------
-- Evaluation
------------------------------------------------------------------------------------------------------------------------------------

trainedConvnet:evaluate()
nTestImages = {1,2,4,8,16,32,64,128}

for n = 1,#nTestImages do
    print('test multiple images '..nTestImages[n])
    -- default method of computing CMC curve
    computeCMC_MeanPool_RNN(dataset,testInds,trainedConvnet,128,nTestImages[n])
end
