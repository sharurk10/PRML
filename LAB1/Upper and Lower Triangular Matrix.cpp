#include <iostream>
using namespace std; 
int main() {
  int n;
  cout << "Enter the order of the square matrix: ";
  cin >> n;
  int A[n][n];
  cout << "Enter the matrix elements:\n";
  for(int i = 0; i < n; i++) {
    for(int j = 0; j < n; j++) {
      cin >> A[i][j];
    }
  }
  bool upper = true;
  bool lower = true;
  for(int i = 0; i < n; i++) {
    for(int j = 0; j < n; j++) {
      // Elements below the main diagonal
      if(i > j && A[i][j] != 0)
        upper = false;
      // Elements above the main diagonal
      if(i < j && A[i][j] != 0)
        lower = false;
    }
  }
  if(upper)
    cout << "The matrix is Upper Triangular." << endl;
  if(lower)
    cout << "The matrix is Lower Triangular." << endl;
  if(!upper && !lower)
    cout << "The matrix is neither Upper Triangular nor Lower Triangular." << endl;
  return 0;
}   