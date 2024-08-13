import Read
import FCM
import Proposed_Data_complexity_aware_RideNN_ensemble_classifier.run
import Classifier_pool_generation.run
import Dynamic_ensemble_algorithm.run
import Filtering_based_oversampling.run
import Neighborhood_based_measures.run



def callmain(dts,tr):

    ACC,TPR,TNR=[],[],[]
    Data_path=Read.folder('Dataset/'+dts+'/*/*tra.arff')

    Data,Label=Read.callmain(Data_path)
    cs=3  # Cluster size
    Cgroup,Clabel=FCM.callamin(Data,Label,cs)
    #--------------------Proposed_Data_complexity_aware_RideNN_ensemble_classifier---------------
    Proposed_Data_complexity_aware_RideNN_ensemble_classifier.run.callmain(Cgroup, Clabel, cs, tr, ACC, TPR, TNR)
    #-----------------Comparative method---------
    Classifier_pool_generation.run.callmain(Cgroup, Clabel, cs, tr, ACC, TPR, TNR)
    Dynamic_ensemble_algorithm.run.callmain(Cgroup, Clabel, cs, tr, ACC, TPR, TNR)
    Filtering_based_oversampling.run.callmain(Cgroup, Clabel, cs, tr, ACC, TPR, TNR)
    Neighborhood_based_measures.run.callmain(Cgroup, Clabel, cs, tr, ACC, TPR, TNR)

    return ACC,TPR,TNR


