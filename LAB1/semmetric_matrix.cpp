#include <iostream>
using namespace std; 
int main() {    
  int n;
  cout << "Enter matrix size: ";
  cin >> n;
  int A[n][n];
  cout << "Enter matrix elements:" << endl;
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < n; j++) {
      cin >> A[i][j];
    }
  }
  cout << "\nInput Matrix:\n";   
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < n; j++) {
      cout << A[i][j] << " ";
    }
    cout << endl;
  }
  bool symmetric = true;
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < n; j++) {
      if (A[i][j] != A[j][i]) {
        symmetric = false;
        break;
      }
    }
    if (!symmetric) {
      break;
        }
    }
    if (symmetric)
        cout << "The matrix is symmetric." << endl;
    else
        cout << "The matrix is not symmetric." << endl;
    return 0;
}