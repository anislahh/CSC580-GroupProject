#include <mpi.h>
#include <iostream>
#include <algorithm>

using namespace std;

#define ROWS 15
#define COLS 3   // employee_id, experience_years, salary_usd

int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);

    int rank = 0, size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // ==============================
    // 2D Dataset Initialization
    // ==============================
    int employees[ROWS][COLS] = {
        {1,  0,  119914},
        {2,  23,  97545},
        {3,  18, 183176},
        {4,  6,   86226},
        {5,  17, 110606},
        {6,  7,  140515},
        {7,  24, 120598},
        {8,  8,   70029},
        {9,  2,  161963},
        {10, 19, 115254},
        {11, 12,  70256},
        {12, 18,  67279},
        {13, 21, 114614},
        {14, 3,  105857},
        {15, 5,   67865}
    };

    // ==============================
    // Work Distribution (static chunks)
    // ==============================
    int chunk = (ROWS + size - 1) / size;        // ceil
    int start = rank * chunk;
    int end   = min(start + chunk, ROWS);        // end exclusive
    bool hasWork = (start < ROWS);

    // ==============================
    // Local Computation
    // ==============================
    int local_max_salary = -1;
    int local_salary_id  = -1;

    int local_max_exp = -1;
    int local_exp_id  = -1;

    if (hasWork) {
        for (int i = start; i < end; i++) {
            int emp_id = employees[i][0];
            int exp    = employees[i][1];
            int salary = employees[i][2];

            if (salary > local_max_salary) {
                local_max_salary = salary;
                local_salary_id  = emp_id;
            }
            if (exp > local_max_exp) {
                local_max_exp = exp;
                local_exp_id  = emp_id;
            }
        }
    }

    // ==============================
    // Print process info ONE-BY-ONE in rank order
    // (and make sure it finishes BEFORE printing results)
    // ==============================
    for (int r = 0; r < size; r++) {
        MPI_Barrier(MPI_COMM_WORLD);

        if (rank == r) {
            if (rank == 0 && r == 0) {
                cout << "=== MPI PARALLEL PROCESSING ===\n";
                cout << "Number of processes: " << size << "\n";
                cout << "Employees: " << ROWS << "\n\n";
            }

            if (hasWork) {
                cout << "Process " << rank << ": Processing employees ["
                     << start << " - " << (end - 1) << "]\n";
                cout << "Process " << rank << ": Local Max Salary = "
                     << local_max_salary << " USD (Employee ID: " << local_salary_id << ")\n";
                cout << "Process " << rank << ": Local Max Experience = "
                     << local_max_exp << " years (Employee ID: " << local_exp_id << ")\n\n";
            } else {
                cout << "Process " << rank << ": No employees assigned\n\n";
            }

            cout.flush();
        }
    }

    // Ensure ALL process outputs are fully flushed BEFORE reductions/results
    MPI_Barrier(MPI_COMM_WORLD);

    // ==============================
    // Global Reduction (MAXLOC)
    // Pair: (value, id)
    // ==============================
    int local_salary_pair[2]  = { local_max_salary, local_salary_id };
    int global_salary_pair[2] = { -1, -1 };

    int local_exp_pair[2]     = { local_max_exp, local_exp_id };
    int global_exp_pair[2]    = { -1, -1 };

    MPI_Reduce(local_salary_pair, global_salary_pair, 1, MPI_2INT, MPI_MAXLOC, 0, MPI_COMM_WORLD);
    MPI_Reduce(local_exp_pair,    global_exp_pair,    1, MPI_2INT, MPI_MAXLOC, 0, MPI_COMM_WORLD);

    // Make sure reductions finished everywhere
    MPI_Barrier(MPI_COMM_WORLD);

    // ==============================
    // Output Result (rank 0 only) - PRINT LAST
    // ==============================
    if (rank == 0) {
        cout << "=== RESULTS ===\n";
        cout << "Highest Salary: " << global_salary_pair[0]
             << " USD (Employee ID: " << global_salary_pair[1] << ")\n";
        cout << "Most Experienced Employee: " << global_exp_pair[0]
             << " years (Employee ID: " << global_exp_pair[1] << ")\n";
        cout.flush();
    }

    // Final barrier so nobody exits early while rank 0 prints
    MPI_Barrier(MPI_COMM_WORLD);

    MPI_Finalize();
    return 0;
}
