#include <iostream> 
#include <vector>
using namespace std; 
int main() {
  int m, n;
  cout << "Enter rows and columns: ";
  cin >> m >> n;
  vector<vector<int>> A(m, vector<int>(n));
  cout << "Enter matrix elements:\n";
  for (int i = 0; i < m; i++) {
    for (int j = 0; j < n; j++) {
      cin >> A[i][j];
    }
  }
  cout << "\nTranspose Matrix:\n";
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < m; j++) {
      cout << A[j][i] << " ";
    }
    cout << endl;
  }
  return 0;
}