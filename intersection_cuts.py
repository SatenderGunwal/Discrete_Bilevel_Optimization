import numpy as np
from mpmath import matrix

class Intersection_Cut:

    def __init__(self, A, b, B, problem_data, HPR_Solution_X, HPR_Solution_Y, Optimal_Y, BasisIndexes ):

        self.A  = problem_data[5]
        self.B  = problem_data[6]
        self.C  = problem_data[7]
        self.Dy = problem_data[8]

        self.Ahat  = np.array(A)
        self.bhat  = np.array(b)
        self.Bhat  = np.array(B)

        self.problem_senses = problem_data[9]
        self.Optimal_Y      = Optimal_Y
        self.HPR_Solution_X = HPR_Solution_X
        self.HPR_Solution_Y = HPR_Solution_Y
        self.BasisIndexes   = BasisIndexes

        self.XDim = len(self.HPR_Solution_X)
        self.YDim = len(self.HPR_Solution_Y)

    # Gives bilevel free set in standard form.
    def Bilevel_Free_Set(self):

        OptY  = self.Optimal_Y
        if OptY == str: return "FMILP Error"

        # RHS 
        DyHat            = self.Dy @ np.array(OptY)
        Bilevel_Free_RHS = self.C + np.ones((len(self.C))) - self.B @ np.array(OptY)
        Final_RHS        = np.array([-1*DyHat] + list(Bilevel_Free_RHS))

        # LHS
        LHS_Top   = np.hstack( (np.zeros(self.XDim), -1*self.Dy) )
        LHS_Lower = np.hstack( (self.A, np.zeros((self.A.shape[0], self.YDim))) ) # Adding zero array block
        Final_LHS = np.vstack( (LHS_Top, LHS_Lower) )

        return Final_LHS, Final_RHS

    # Defines the intersection cut.
    def Cut(self):

        Num_Vars = self.XDim + self.XDim
        Final_LHS, Final_RHS = self.Bilevel_Free_Set()

        try:
            B_hat_pseudoinverse = matrix(self.Bhat)**-1
            B_hat_pseudoinverse = np.array(B_hat_pseudoinverse.tolist())
        except:
            print("\nError in Inverse (Singular Matrix)...\n")
            print(f"\nBasic Variable Indexes {len(self.BasisIndexes)} = ", self.BasisIndexes)
            print(f"\nBHat Matrix {self.Bhat.shape} = ", self.Bhat)

        # Code Line 1-3 from Algorithm 1(IC Separation) from Fischetti
        num_disjunctions = Final_LHS.shape[0]
        for row_num in range(num_disjunctions):
            older_lhs = np.array([list(Final_LHS[row_num])])#Final_LHS[row_num]
            older_rhs = Final_RHS[row_num]
            U_i       = np.take(older_lhs, self.BasisIndexes, axis=1) @ B_hat_pseudoinverse
            Final_LHS[row_num] = older_lhs - U_i @ self.Ahat
            Final_RHS[row_num] = older_rhs - U_i @ self.bhat

        # Code Line 4
        gamma = []
        for idx in range( Num_Vars ):
            column_   = np.take(Final_LHS, idx,axis=1)
            coef_list = [ column_[i]/Final_RHS[i] for i in range(num_disjunctions) ]
            gamma.append(max(coef_list))

        # Code line 5-9 (Skipping Line 7 by assuming all variables are integers)
        if all(x >= 0 for x in gamma):
            for i in range( Num_Vars ):
                gamma[i] = min([gamma[i],1])
        # Code line 10
        return gamma