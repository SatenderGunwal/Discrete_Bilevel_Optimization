from pyscipopt import Model, SCIP_PARAMSETTING, quicksum
import numpy as np
import pickle as pkl
from constraint_handler import Constraint_Handler
# import os

def GBP_CUT( problem_data ):

    [Cx, Cy, Gx, Gy, G0, A, B, C, Dy, problem_senses ] = problem_data
    XDim, YDim                                         = len(Cx), len(Dy)

    model = Model("General Binary Bilevel Solver")
    model.setPresolve(SCIP_PARAMSETTING.OFF)
    model.setHeuristics(SCIP_PARAMSETTING.OFF)
    

    # model.hideOutput(False)
    model.setParam('limits/time', 3600)

    X = { f"{i}":model.addVar(vtype="B", name=f"X{i}") for i in range(XDim) }
    Y = { f"{i}":model.addVar(vtype="B", name=f"Y{i}") for i in range(YDim) }

    # Objective
    model.setObjective( quicksum( Cx[idx]*X[f'{idx}'] for idx in range(XDim) ) + quicksum( Cy[idx]*Y[f'{idx}'] for idx in range(YDim) ), 'minimize' )

    # Original Leader's and Follower's Constraints
    num_leader_cons   = len(Gx)
    num_follower_cons = len(A)
    for row in range(num_leader_cons):
        model.addCons( quicksum( Gx[row][idx]*X[f'{idx}'] for idx in range(XDim)) + quicksum( Gy[row][idx]*Y[f'{idx}'] for idx in range(YDim)) <= G0[row], name='ULC' )
    for row in range(num_follower_cons):
        model.addCons( quicksum( A[row][idx]*X[f'{idx}'] for idx in range(XDim)) + quicksum( B[row][idx]*Y[f'{idx}'] for idx in range(YDim)) <= C[row], name='LLC' )

    #model.data = X,Y,PIx,PIy,V,PIu
    conshdlr = Constraint_Handler(X,Y)
    # conshdlr.data = XDim, YDim, Dy, A, B, C
    conshdlr.data = problem_data, XDim, YDim
    model.includeConshdlr(conshdlr,  "GBP", "Constraint handler for GBP", needscons = False) # needscons: "should the constraint handler be skipped, if no constraints are available?"

    model.optimize()
    obj = model.getObjVal()

    return obj

if __name__=="__main__":

    results = []
    Num_Tests = 1
    for test in range(Num_Tests):
            for id_ in range(2,3):

                file_path = ".\\GBP_10_10_5_5\\GBP_10_10_5_5_1.pkl"
                # file_path = os.path.join("GBP_10_10_5_5", "GBP_10_10_5_5_1.pkl")
                # print("File path = ", file_path)
                with open(file_path, 'rb') as file:
                    data = pkl.load(file)

                # data = pkl.load(open(file_path, 'rb'))
                # data = pkl.load(open(f'/content/drive/MyDrive/Binary_Bilevel_Solver/Demo_Datasets/For_SCIP/GBP_4_4_2_2_4.pkl', 'rb'))

                # Changing problem to Min-Min
                [Cx, Cy, Gx, Gy, G0, A, B, C, Dy, senses] = data

                # HPR Constraint Matrix Rank Check
                B1 = np.hstack((Gx,Gy))
                B2 = np.hstack((A,B))
                M = np.vstack((B1,B2))
                rank = np.linalg.matrix_rank(M)
                print("Rank of HPR Matrix = ", rank)

                # print(senses, Cx, Cy )
                if senses[0] == -1:
                    Cx, Cy = -Cx, -Cy
                    senses[0] = 1
                if senses[1] == -1:
                    Dy = -Dy
                    senses[1] = 1
                data = [Cx, Cy, Gx, Gy, G0, A, B, C, Dy, senses]
                #----------------------------------------------
                obj = GBP_CUT(data)
                print(obj)