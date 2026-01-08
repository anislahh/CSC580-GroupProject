#include <iostream>
#include <omp.h>

using namespace std;

#define ROWS 15
#define COLS 6
/*
  We store the dataset as ONE 2D integer array (ROWS x COLS).

  Column meaning (index):
  0 = employee_id
  1 = experience_years
  2 = salary_usd
  3 = industry_code      (encoded as numbers)
  4 = location_code      (encoded as numbers)
  5 = education_code     (encoded as numbers)

  Why encode strings into numbers?
  - Because int[][] can only store integers (fast + simple for parallel loops).
*/

static const char* industryName(int code) {
    // Convert industry_code -> readable text (for printing)
    switch (code) {
        case 0: return "Finance";
        case 1: return "Healthcare";
        case 2: return "Technology";
        case 3: return "Consulting";
        case 4: return "Retail";
        case 5: return "Education";
        default: return "Unknown";
    }
}

static const char* locationName(int code) {
    // Convert location_code -> readable text (for printing)
    switch (code) {
        case 0: return "Urban";
        case 1: return "Rural";
        case 2: return "Suburban";
        default: return "Unknown";
    }
}

static const char* eduName(int code) {
    // Convert education_code -> readable text (for printing)
    switch (code) {
        case 0: return "High School";
        case 1: return "Bachelor";
        case 2: return "Master";
        case 3: return "PhD";
        default: return "Unknown";
    }
}

/*
  Helper function: print one employee row in a clean format.
*/
static void printEmployee(const char* title, const int data[ROWS][COLS], int rowIndex) {
    cout << title << "\n";
    if (rowIndex < 0) return; // safety: in case no valid row was found

    cout << "Employee ID: " << data[rowIndex][0] << "\n";
    cout << "Industry   : " << industryName(data[rowIndex][3]) << "\n";
    cout << "Experience : " << data[rowIndex][1] << " years\n";
    cout << "Salary     : " << data[rowIndex][2] << " USD\n";
    cout << "Location   : " << locationName(data[rowIndex][4]) << "\n";
    cout << "Education  : " << eduName(data[rowIndex][5]) << "\n";
}

int main() {
    /*
      Dataset (15 employees).
      All text fields are encoded into numbers (industry/location/education).
      Later, we decode them back into words ONLY when printing.
    */
    int data[ROWS][COLS] = {
        { 1,  0, 119914, 0, 0, 2},
        { 2, 23,  97545, 1, 1, 0},
        { 3, 18, 183176, 0, 0, 3},
        { 4,  6,  86226, 0, 1, 1},
        { 5, 17, 110606, 2, 1, 1},
        { 6,  7, 140515, 3, 1, 3},
        { 7, 24, 120598, 2, 1, 1},
        { 8,  8,  70029, 4, 0, 1},
        { 9,  2, 161963, 0, 0, 3},
        {10, 19, 115254, 4, 0, 2},
        {11, 12,  70256, 0, 1, 0},
        {12, 18,  67279, 4, 1, 1},
        {13, 21, 114614, 2, 2, 1},
        {14,  3, 105857, 5, 2, 3},
        {15,  5,  67865, 4, 2, 2}
    };

    // =========================================================
    // STEP 1: Find how many OpenMP threads are ACTUALLY used
    // =========================================================
    /*
      OpenMP can use multiple threads. A "thread" is basically a worker.
      If you set OMP_NUM_THREADS=8, you should get 8 threads.

      We enter a small parallel region and ask OpenMP:
      "How many threads did you create?"
    */
    int nt_used = 1;
    #pragma omp parallel
    {
        // "single" means: only ONE thread executes this line (to avoid duplicates)
        #pragma omp single
        nt_used = omp_get_num_threads();
    }

    cout << "OpenMP Threads (used): " << nt_used << "\n";
    cout << "Employees: " << ROWS << "\n\n";

    // =========================================================
    // STEP 2: Show which thread processes which employees
    // =========================================================
    cout << "=== Thread Work Assignment ===\n";

    // starts[t] and ends[t] store the range of indices handled by thread t
    int starts[256], ends[256];
    for (int i = 0; i < 256; i++) { starts[i] = 0; ends[i] = -1; }

    /*
      We compute a "chunk size" = how many rows each thread should take.
      chunk = ceil(ROWS / nt_used)
      Example: ROWS=15, threads=8 -> chunk=2
      - Thread 0 gets rows 0..1
      - Thread 1 gets rows 2..3
      - Last thread might get fewer rows (example: only row 14)
    */
    #pragma omp parallel
    {
        int tid = omp_get_thread_num();

        int chunk = (ROWS + nt_used - 1) / nt_used; // ceil division
        int start = tid * chunk;
        int end   = start + chunk - 1;

        // Clamp the end so we don't go beyond last row
        if (end >= ROWS) end = ROWS - 1;

        starts[tid] = start;
        ends[tid]   = end;
    }

    // Print in order (0..nt_used-1) so output looks neat and easy to read
    for (int t = 0; t < nt_used; t++) {
        if (starts[t] < ROWS && ends[t] >= starts[t]) {
            cout << "Thread " << t << ": Processing employees ["
                 << starts[t] << " - " << ends[t] << "]\n";
        } else {
            cout << "Thread " << t << ": No employees assigned\n";
        }
    }
    cout << "\n";

    // =========================================================
    // OUTPUT 1: Highest Salary Employee (parallel max)
    // =========================================================
    /*
      Goal: find the employee with the highest salary.

      Strategy:
      - Each thread scans part of the array and finds its local best salary.
      - Then we combine all local results into ONE global best salary.

      Why do this in parallel?
      - For big datasets, splitting the scan saves time.
      - For small datasets like 15 rows, it still demonstrates the concept.
    */
    double t1 = omp_get_wtime(); // start timer

    int bestSalary = -1;      // global best salary value
    int bestSalaryRow = -1;   // which row has that best salary

    #pragma omp parallel
    {
        int localBest = -1;   // best salary found by THIS thread
        int localRow  = -1;   // row index for that local best

        // Each thread runs a slice of the for-loop
        #pragma omp for schedule(static) nowait
        for (int i = 0; i < ROWS; i++) {
            int s = data[i][2]; // salary column
            if (s > localBest) {
                localBest = s;
                localRow = i;
            }
        }

        /*
          Multiple threads will try to update the global best.
          "critical" means only one thread can enter at a time,
          so we don't get race conditions (data corruption).
        */
        #pragma omp critical
        {
            if (localBest > bestSalary) {
                bestSalary = localBest;
                bestSalaryRow = localRow;
            }
        }
    }

    double t2 = omp_get_wtime(); // end timer

    printEmployee("=== Output 1: Highest Salary Employee ===", data, bestSalaryRow);
    cout << "Runtime (Output 1): " << (t2 - t1) << " seconds\n\n";

    // =========================================================
    // OUTPUT 2: Most Experienced Employee (parallel max)
    // =========================================================
    double t3 = omp_get_wtime(); // start timer

    int bestExp = -1;      // global best experience value
    int bestExpRow = -1;   // which row has that best experience

    #pragma omp parallel
    {
        int localBest = -1;
        int localRow  = -1;

        #pragma omp for schedule(static) nowait
        for (int i = 0; i < ROWS; i++) {
            int e = data[i][1]; // experience column
            if (e > localBest) {
                localBest = e;
                localRow = i;
            }
        }

        #pragma omp critical
        {
            if (localBest > bestExp) {
                bestExp = localBest;
                bestExpRow = localRow;
            }
        }
    }

    double t4 = omp_get_wtime(); // end timer

    printEmployee("=== Output 2: Most Experienced Employee ===", data, bestExpRow);
    cout << "Runtime (Output 2): " << (t4 - t3) << " seconds\n\n";
    /*
      Employee ID 15 is at row index 14 (because arrays start from 0).
      We check which thread's assigned range includes index 14.
    */
    int employee15_index = 14; // row index for employee_id 15
    int ownerThread = -1;

    for (int t = 0; t < nt_used; t++) {
        if (starts[t] <= employee15_index && employee15_index <= ends[t]) {
            ownerThread = t;
            break;
        }
    }

    cout << "Info: Employee ID 15 (index 14) is processed by Thread " << ownerThread << ".\n";

    return 0;
}
