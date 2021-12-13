import time
import math
import numpy as np
import argparse
from scipy.sparse import csr_matrix
from scipy.spatial.distance import jaccard,cosine



# creates sparse matrixes
def load_and_sparse(path,measure):
    data = np.load(path)
    if measure == 'js':
        return csr_matrix((np.full(data.shape[0],1), (data[:,1],data[:,0])))
    if measure == 'cs':
        return csr_matrix((data[:,2], (data[:,0],data[:,1])))
    if measure == 'dcs':
        return csr_matrix((np.full(data.shape[0],1), (data[:,0],data[:,1])))



# creates projection matrix for cs and dcs
def get_projections(n_projections,csr_shape):
    mu, sigma = 0, 1
    vectors = np.random.normal(mu,sigma,(n_projections,csr_shape[1]))
    return vectors/np.linalg.norm(vectors)



# calculates approximate similarity on the signature matrix
# to find candidate pairs.
# We take the number of same items in the same position and
# we divide it by the total length of the signature
def approx_similarity(usr_1, usr_2, signature_matr, signature_len):
    return float(np.count_nonzero(signature_matr[usr_1] == signature_matr[usr_2])
                ) / signature_len



#COSINE AND DISCRETE COSINE SIMILARITY MAIN FUNCTION
def get_cosine_pairs(path, measure, n_projections, n_bands, threshold=0.73, debug=True):

    #create sparse matrix
    csr_sparse = load_and_sparse(path,measure)
    csr_shape = csr_sparse.get_shape()
    if debug:
        print("***********************************************************")
        print(f"csr matrix shape: {csr_shape}, non zero elements: {csr_sparse.count_nonzero()}")

    # create projections
    vectors = get_projections(n_projections,csr_shape)
    if debug:
        print("***********************************************************")
        print("projections table shape:", vectors.shape)

    # create signature matrix
    start_time = time.time()
    signature_matrix = np.array([csr_sparse.dot(vectors[j])
                            for j in range(len(vectors))]).T
    signature_matrix[ signature_matrix > 0] = 1
    signature_matrix[ signature_matrix <= 0] = -1
    end_time = time.time()
    if debug:
        print("***********************************************************")
        print("signature matrix shape:",signature_matrix.shape)
        print(f"time to create signature_matrix: {round((end_time - start_time),2)}")

    # bands creation
    sig_bands = [ math.floor(n_projections/n_bands)*i for i in range(n_bands+1) ]

    # candidate pairs calculation
    if debug:
        print("***********************************************************")
        print(f"searching candidate pairs for {n_bands} bands: and threshold of {threshold}...")

    start_time = time.time()
    candidate_pairs = set()
    range_curr_band = range(csr_shape[0])
    range_sig_bands = range(len(sig_bands)-1)
    # loop for each band
    for curr_band_idx in range_sig_bands:
        curr_band = signature_matrix[:,sig_bands[curr_band_idx]:sig_bands[curr_band_idx+1]]
        dict_of_hashes = {}
        # loop for each user in band and add to dictionary of hashes
        for curr_usr_idx in range_curr_band:
            curr_hash = hash(tuple(curr_band[curr_usr_idx]))
            dict_of_hashes.setdefault(curr_hash, []).append(curr_usr_idx)

        # add pairs from dictionary (buckets) to candidate pairs
        tot_comparisons = 0
        for curr_key in dict_of_hashes:
            if (len(dict_of_hashes[curr_key]) < 150) and (len(dict_of_hashes[curr_key]) > 1):
                curr_list = dict_of_hashes[curr_key]
                range_curr_list = range(len(curr_list)-1)
                #loop all pairs in each bucket
                for i in range_curr_list:
                    second_rng = range(i+1,len(curr_list))
                    for j in second_rng:
                        # add pair only if signature similarity is bigger than the threshold
                        if approx_similarity(curr_list[i],curr_list[j],signature_matrix,n_projections) > threshold:
                            new_pair = (curr_list[i], curr_list[j])
                            candidate_pairs.add(new_pair)
                        tot_comparisons += 1

    #sort user pairs by smaller id first.
    candidate_pairs = sorted(candidate_pairs, key=lambda x: x[0])

    end_time = time.time()
    time_get_all = end_time - start_time
    if debug:
        print(f"total approximate comparisons: {tot_comparisons}, candidate pairs:{len(candidate_pairs)}")
        print(f"time to calculate all pairs: {round(time_get_all,2)}")

    # calculate cosine similarity and write to file
    if measure == 'dcs':
        file_name = 'dcs.txt'
    else:
        file_name = 'cs.txt'

    start_time = time.time()
    # override old file
    file_out = open(file_name, "w")
    file_out.close()
    counter = 0
    for i in candidate_pairs:
        curr_score = 1 - (math.acos(1 - cosine(csr_sparse[i[0]].toarray(),csr_sparse[i[1]].toarray()))*(180/math.pi))/180
        if curr_score >= threshold:
            counter += 1
            # append new pair
            file_out = open(file_name, "a")
            file_out.write(str(i[0])+","+" "+str(i[1])+"\n")
            file_out.close()
    
    end_time = time.time()
    if debug:
        print("***********************************************************")
        print("actual pairs found:",counter)
        print(f"time to calculate actual pairs: {round((end_time - start_time),2)}")




#JACCARD SIMILARITY MAIN FUNCTION
def get_jaccard_pairs(path, measure, n_permutations=120, n_bands=20, threshold=0.5, debug=True):
    
    # sparse matrix 
    csr_sparse = load_and_sparse(path,measure)
    csr_shape = csr_sparse.shape
    if debug:
        print("***********************************************************")
        print(f"csr matrix shape: {csr_shape}, non zero elements: {csr_sparse.count_nonzero()}")

    # hashes / permutations of rows
    row_permutations = np.array([ np.random.permutation(csr_shape[0]) 
                                    for i in range(n_permutations)]).T
    if debug:
        print("***********************************************************")
        print("permutations table shape:", row_permutations.shape)

    # signature matrix creation
    start_time = time.time()
    signature_matrix = np.full((csr_shape[1], n_permutations), np.inf)
    # signature updates
    csr_range_rows = range(csr_shape[0])
    for curr_csr_row in csr_range_rows:
        non_zero_col_indexes = csr_sparse[curr_csr_row].nonzero()[1]
        curr_permut_vals = row_permutations[curr_csr_row]

        for curr_col in non_zero_col_indexes:
            signature_matrix[curr_col] = np.minimum(signature_matrix[curr_col],
                                                curr_permut_vals)
    end_time = time.time()
    if debug:
        print("***********************************************************")
        print("signature matrix shape:",signature_matrix.shape)
        print(f"time to create signature_matrix: {round((end_time - start_time),2)}")

    # calculate indexes of bands
    sig_bands = [ math.floor(n_permutations/n_bands)*i for i in range(n_bands+1) ]

    # calculation of candidate pairs
    if debug:
        print("***********************************************************")
        print(f"searching candidate pairs for {n_bands} bands and threshold of {threshold}...")
    start_time = time.time()

    pairs = set()
    range_curr_band = range(csr_shape[1])
    range_sig_bands = range(len(sig_bands)-1)
    # loop for each band
    for curr_band_idx in range_sig_bands:

        curr_band = signature_matrix[:,sig_bands[curr_band_idx]:sig_bands[curr_band_idx+1]]
        dict_of_hashes = {}
        # loop for each user in band and add to dictionary of hashes
        for curr_usr_idx in range_curr_band:
            curr_hash = hash(tuple(curr_band[curr_usr_idx]))
            dict_of_hashes.setdefault(curr_hash, []).append(curr_usr_idx)

        # add pairs to array from dict 
        # for each pair of users in our dictionary 
        # we calculate their approximate similarity
        # and add them to our candidate pairs if similarity > 0.5
        tot_comparisons = 0
        for curr_key in dict_of_hashes:
            # take only buckets with more than 1 item
            if len(dict_of_hashes[curr_key]) > 1:
                curr_list = dict_of_hashes[curr_key]
                range_curr_list = range(len(curr_list)-1)
                #loop all pairs in each bucket
                for i in range_curr_list:
                    second_rng = range(i+1,len(curr_list))
                    for j in second_rng:
                        # add pair only if signature similarity is bigger than the threshold
                        if approx_similarity(curr_list[i],curr_list[j],signature_matrix,n_permutations) > threshold:
                            new_pair = (curr_list[i], curr_list[j])
                            pairs.add(new_pair)
                        tot_comparisons += 1
    pairs = sorted(pairs, key=lambda x: x[0])

    end_time = time.time()
    time_get_all = end_time - start_time
    if debug:
        print(f"total approximate comparisons: {tot_comparisons}, candidate pairs:{len(pairs)}")
        print(f"time to calculate all pairs: {round(time_get_all,2)}")

    # calculate jaccard and add pairs
    start_time = time.time()
    csr_sparse_t = csr_sparse.tocsc()
    file_out = open("js.txt", "w")
    file_out.close()
    counter = 0
    for i in pairs:
        # we can use the jaccard distance function of scipy since our domain is just [0,1]
        curr_score = 1 - jaccard(csr_sparse_t[:,i[0]].toarray(),csr_sparse_t[:,i[1]].toarray())
        if curr_score >= 0.5:
            counter += 1
            file_out = open("js.txt", "a")
            file_out.write(str(i[0])+","+" "+str(i[1])+"\n")
            file_out.close()
    end_time = time.time()
    if debug:
        print("***********************************************************")
        print("actual pairs found:",counter)
        print(f"time to calculate actual jaccard similarities: {round((end_time - start_time),2)}")




def main():
    parser = argparse.ArgumentParser()
    # main arguments required for the assignment
    parser.add_argument("-d", action="store", dest="dir", default='user_movie_rating.npy', type=str)
    parser.add_argument("-s", action="store", dest="seed", default=0, type=int)
    parser.add_argument("-m", action="store", dest="measure", default='js', type=str)
    # extra arguments for experimentation
    parser.add_argument("-b", action="store", dest="bands", 
                        help="Can be used to set the number of bands",
                        type=int)
    parser.add_argument("-p", action="store", dest="permutations", 
                        help="Can be used to set the signature size",
                         type=int)
    parser.add_argument("-t", action="store", dest="threshold", 
                        help="Can be used to set the threshold",
                        type=float)
    parser.add_argument("-v", action="store", dest="verbose", 
                        help="Can be set to 1 to see debug messages on terminal",
                        default=0, type=int)
    args = parser.parse_args()
    print("arguments passed:",args)

    # set global params
    np.random.seed(args.seed)
    debug=args.verbose

    global_start_time = time.time()
    if args.measure == 'js':
        # params for jaccard if not specified
        permutations = args.permutations or 120
        bands = args.bands or 20
        threshold = args.threshold or 0.5
        if debug:
            print()
            print("***********************************************************")
            print(f'main params:')
            print(f"signature size: {permutations}, bands: {bands}, threshold: {threshold}")


        # main function for jaccard siilarity
        get_jaccard_pairs(args.dir, args.measure, permutations, bands, threshold, debug)

    elif args.measure == 'cs' or args.measure == 'dcs':
        # params for cosine similarity and discrete cosine similarity if not specified
        permutations = args.permutations or 150
        bands = args.bands or 10
        threshold = args.threshold or 0.73
        if debug:
            print()
            print("***********************************************************")
            print(f'main params:')
            print(f"signature size: {permutations}, bands: {bands}, threshold: {threshold}")
        
        # main function for cs and dcs
        get_cosine_pairs(args.dir, args.measure, permutations, bands, threshold, debug)

    global_end_time = time.time()
    if debug:
        print("***********************************************************")
        print(f'total runtime: {round(global_end_time - global_start_time,2)}')
        print("***********************************************************")

if __name__ == "__main__":
    main()