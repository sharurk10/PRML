#include <iostream>
using namespace std;
int main() {
  int n;
  cout << "Enter vector size: ";
  cin >> n;
  int A[n], B[n];
  int dot = 0;
  cout << "Enter first vector:\n";
  for (int i = 0; i < n; i++)
    cin >> A[i];
  cout << "Enter second vector:\n";
  for (int i = 0; i < n; i++)
    cin >> B[i];
  for (int i = 0; i < n; i++)
    dot += A[i] * B[i];
  cout << "Dot Product = " << dot;
  return 0;
}