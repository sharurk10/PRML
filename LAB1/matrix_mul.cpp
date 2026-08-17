#include <iostream>
using namespace std;
int main() {
  int m, n, k;
  cout << "enter rows of matrix A: ";
  cin >> m;
  cout << "enter columns of matrix A: ";
  cin >> n;
  cout << "enter columns of matrix B: ";
  cin >> k;
  int A[m][n], B[n][k], C[m][k];
  cout << "enter matrix A:" << endl;
  for (int i = 0; i < m; i++)
    for (int j = 0; j < n; j++)
      cin >> A[i][j];
  cout << "enter matrix B:" << endl;
  for (int i = 0; i < n; i++)
    for (int j = 0; j < k; j++)
      cin >> B[i][j];
  for (int i = 0; i < m; i++) {
    for (int j = 0; j < k; j++) {
      C[i][j] = 0;
      for (int l = 0; l < n; l++)
        C[i][j] += A[i][l] * B[l][j];
    }
  }
  cout << "Result:" << endl;
  for (int i = 0; i < m; i++) {
    for (int j = 0; j < k; j++)
      cout << C[i][j] << " ";
    cout << endl;
  }
  return 0;
}