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

function buildModel_MeanPool_RNN(opt, nFltrs1,nFltrs2,nFltrs3,nPersonsTrain)

    local nFilters = {nFltrs1,nFltrs2,nFltrs3}

    local filtsize = {5,5,5}
    local poolsize = {2,2,2}
    local stepSize = {2,2,2}

    -- remember this adds padding to ALL SIDES of the image
    local padDim = 4

    local cnn = nn.Sequential()

    local ninputChannels = 5
    cnn:add(nn.SpatialZeroPadding(padDim, padDim, padDim, padDim))
    cnn:add(nn.SpatialConvolutionMM(ninputChannels, nFilters[1], filtsize[1], filtsize[1], 1, 1))
    cnn:add(nn.Tanh())
    cnn:add(nn.SpatialMaxPooling(poolsize[1],poolsize[1],stepSize[1],stepSize[1]))

    ninputChannels = nFilters[1]
    cnn:add(nn.SpatialZeroPadding(padDim, padDim, padDim, padDim))
    cnn:add(nn.SpatialConvolutionMM(ninputChannels, nFilters[2], filtsize[2], filtsize[2], 1, 1))
    cnn:add(nn.Tanh())
    cnn:add(nn.SpatialMaxPooling(poolsize[2],poolsize[2],stepSize[2],stepSize[2]))

    ninputChannels = nFilters[2]
    cnn:add(nn.SpatialZeroPadding(padDim, padDim, padDim, padDim))
    cnn:add(nn.SpatialConvolutionMM(ninputChannels, nFilters[3], filtsize[3], filtsize[3], 1, 1))
    cnn:add(nn.Tanh())
    local nFullyConnected
    if opt.spatial == 1 then
        cnn:add(inn.SpatialPyramidPooling({8,8},{4,4},{2,2},{1,1}))
        nFullyConnected = 32*(64+16+4+1)
    else
        cnn:add(nn.SpatialMaxPooling(poolsize[3],poolsize[3],stepSize[3],stepSize[3]))
        nFullyConnected = nFilters[3]*10*8    
    end    

    cnn:add(nn.Reshape(1,nFullyConnected))
    cnn:add(nn.Dropout(opt.dropoutFrac))    
    cnn:add(nn.Linear(nFullyConnected,opt.embeddingSize))
    cnn:cuda()

    local h2h = nn.Sequential()
    h2h:add(nn.Tanh())
    h2h:add(nn.Dropout(opt.dropoutFracRNN))
    h2h:add(nn.Linear(opt.embeddingSize,opt.embeddingSize))
    h2h:cuda()

    local r1 = nn.Recurrent(
        opt.embeddingSize,
        cnn,
        h2h,
        nn.Identity(),
        opt.sampleSeqLength)

    local rnn1 = nn.Sequencer(
        nn.Sequential()
        :add(r1)
        )

    Combined_CNN_RNN_1 = nn.Sequential()
    Combined_CNN_RNN_1:add(rnn1)
    Combined_CNN_RNN_1:add(nn.JoinTable(1))
    --Combined_CNN_RNN_1:add(nn.Mean(1))

    local r2 = nn.Recurrent(
        opt.embeddingSize,
        cnn:clone('weight','bias','gradWeight','gradBias'),
        h2h:clone('weight','bias','gradWeight','gradBias'),
        nn.Identity(),
        opt.sampleSeqLength)

    local rnn2 = nn.Sequencer(
        nn.Sequential()
        :add(r2)
        )

    Combined_CNN_RNN_2 = nn.Sequential()
    Combined_CNN_RNN_2:add(rnn2)
    Combined_CNN_RNN_2:add(nn.JoinTable(1))
    --Combined_CNN_RNN_2:add(nn.Mean(1))

    -- Combined_CNN_RNN_2 = Combined_CNN_RNN_1:clone('weight','bias','gradWeight','gradBias')

    local mlp2 = nn.ParallelTable()
    mlp2:add(Combined_CNN_RNN_1)
    mlp2:add(Combined_CNN_RNN_2)
    mlp2:cuda()

    ---------------------------------- attention model start -----------------------------------

    calculate1 = nn.ParallelTable()
    calculate1:add(nn.MetrixMultiply(opt.embeddingSize))
    calculate1:add(nn.Transpose({1,2}))

    calculate2 = nn.ConcatTable()
    calculate2:add(nn.MM())

    calculate = nn.Sequential()
    calculate:add(calculate1)
    calculate:add(calculate2)
    calculate:add(nn.SelectTable(1))
    calculate:add(nn.Tanh())

    attention1 = nn.ConcatTable()
    attention1:add(calculate)
    attention1:add(nn.SelectTable(1))
    attention1:add(nn.SelectTable(2))

    probe_seq = nn.Sequential()
    probe_seq:add(nn.SelectTable(1))
    probe_seq:add(nn.Max(2))
    probe_seq:add(nn.SoftMax())
    probe_seq:add(nn.Unsqueeze(1))
    
    probe1 = nn.ConcatTable()
    probe1:add(probe_seq)
    probe1:add(nn.SelectTable(2))
    probe1:cuda()

    probe5 = nn.ConcatTable()
    probe5:add(nn.MM())
    probe5:cuda()

    gallery_seq = nn.Sequential()
    gallery_seq:add(nn.SelectTable(1))
    gallery_seq:add(nn.Max(1))
    gallery_seq:add(nn.SoftMax())
    gallery_seq:add(nn.Unsqueeze(1))

    gallery1 = nn.ConcatTable()
    gallery1:add(gallery_seq)
    gallery1:add(nn.SelectTable(3))
    gallery1:cuda()

    gallery5 = nn.ConcatTable()
    gallery5:add(nn.MM())
    gallery5:cuda()

    probe = nn.Sequential()
    probe:add(probe1)
    probe:add(probe5)
    probe:cuda()

    gallery = nn.Sequential()
    gallery:add(gallery1)
    gallery:add(gallery5)
    gallery:cuda()

    attention4 = nn.ConcatTable()
    attention4:add(probe)
    attention4:add(gallery)
    attention4:cuda()

    attention5 = nn.ParallelTable()
    attention5:add(nn.SelectTable(1))
    attention5:add(nn.SelectTable(1))
    attention5:cuda()

    local attention = nn.Sequential()
    attention:add(mlp2)
    attention:add(attention1)
    attention:add(attention4)
    attention:add(attention5)

    ---------------------------------- attention model end -----------------------------------

    local mlp3 = nn.ConcatTable()
    mlp3:add(nn.Identity())
    mlp3:add(nn.Identity())
    mlp3:add(nn.Identity())
    mlp3:cuda()

    local mlp4 = nn.ParallelTable()
    mlp4:add(nn.Identity())
    mlp4:add(nn.SelectTable(1))
    mlp4:add(nn.SelectTable(2))
    mlp4:cuda()

    -- used to predict the identity of each person
    local classifierLayer = nn.Linear(opt.embeddingSize,nPersonsTrain)

    -- identification
    local mlp6 = nn.Sequential()
    mlp6:add(classifierLayer)
    mlp6:add(nn.LogSoftMax())
    mlp6:cuda()
    
    local mlp7 = nn.Sequential()
    mlp7:add(classifierLayer:clone('weight','bias','gradWeight','gradBias'))
    mlp7:add(nn.LogSoftMax())
    mlp7:cuda()

    local mlp5 = nn.ParallelTable()
    mlp5:add(nn.PairwiseDistance(2))
    mlp5:add(mlp6)
    mlp5:add(mlp7)
    mlp5:cuda()

    local fullModel = nn.Sequential()
    fullModel:add(attention)
    fullModel:add(mlp3)
    fullModel:add(mlp4)
    fullModel:add(mlp5)
    fullModel:cuda()

    local crit = nn.SuperCriterion()
    crit:add(nn.HingeEmbeddingCriterion(opt.hingeMargin),1)
    crit:add(nn.ClassNLLCriterion(),1)
    crit:add(nn.ClassNLLCriterion(),1)

    return fullModel, crit, attention, cnn
end
