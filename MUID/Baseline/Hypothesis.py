import copy
class Hypothesis:
    """
    Hypothesis which include a track list and the probability of this hypothesis.

    Parameters
    -------------
    tracks: list Track
        use the track list to initialize.

    prob: float
        initialized probability.
    """
    def __init__(self,tracks=[],prob=1):
        self.tracks=tracks
        self.prob=prob

    def __str__(self):
        return '-----------\nlen(tracks)={0}\nprob={1}'\
            .format(len(self.tracks),self.prob)
        
    def Predict(self):
        # print('hypo.preidicting')
        for track in self.tracks:
            track.PredictNextPos()

    def CalHypothesisMat(self,newTracks):
        # print('   calhypothesisMat')
        candidateAssociationTrackIDs=[]
        for newid,newTrack in enumerate(newTracks):
            candidate=[0]
            now=newTrack.trackPoint[-1]
            for idx,track in enumerate(self.tracks):
                if track.ingate(now,15):
                    candidate.append(idx+1)     #选出候选的已有路径索引，添加至candidate
            candidate.append(newid+len(self.tracks)+1)   #将新路径的Id添加到candidate
            candidateAssociationTrackIDs.append(candidate)    #关联矩阵
        # print('len(candidateAssociationTrackIDs)',len(candidateAssociationTrackIDs))
        # print('candidateAssociationTrackIDs',candidateAssociationTrackIDs)
        ans=[]
        def dfs(d,candidate,usedTrackIDs,now):
            #print('   d',d)
            if d==len(candidate):
                ans.append(copy.deepcopy(now))
                return
            for item in candidate[d]:
                if item==0 or item not in usedTrackIDs:
                    if item!=0:
                        usedTrackIDs.add(item)
                    now.append(item)
                    dfs(d+1,candidate,usedTrackIDs,now)
                    now.pop()
                    if item!=0:
                        usedTrackIDs.remove(item)
        dfs(0,candidateAssociationTrackIDs,set(),[])
        return ans

def similar(hyp,hyp_):
    if len(hyp.tracks)!=len(hyp_.tracks):
        return False
    for track in hyp_.tracks:
        flag=False
        for t in hyp.tracks:
            if t.ingate(track.trackPoint[-1]):
                flag=True
                break
        if not flag:
            return False
    return True

def MergeHyps(hyps):
    vis=[False]*len(hyps)
    ret=[]
    se=set(range(1,len(hyps)))
    for i,hyp in enumerate(hyps):
        if vis[i]:
            continue
        se_=copy.deepcopy(se)
        for j in se:
            if similar(hyp,hyps[j]):
                vis[j]=True
                hyp.prob+=hyps[j].prob
                se_.remove(j)
        se=copy.deepcopy(se_)
        ret.append(hyp)
    return ret

def PruneHyps(hyps,num):
    return hyps[:min(num,len(hyps))]

def PruneHypsWithMaxTracks(hyps,maxTracks):
    return [hyp for hyp in hyps if len(hyp.tracks)<=maxTracks]
            
def NormalizeWeight(hyps):
    sumWeight=sum([item.prob for item in hyps])
    
    for hyp in hyps:
        hyp.prob/=sumWeight

def SortHypothesis(hyps):
    hyps.sort(key=lambda x:x.prob,reverse=True)