#include <iostream>
#include <fstream>

using namespace std;

int main() {

  int n, m;
  ifstream myfile;
  myfile.open ("matrix.txt");
  myfile >> n >> m;
  char mat[n][m];

  for (int i = 0; i < n; i++) {
    for (int j = 0; j < m; j++) {
      myfile >> mat[i][j];
    }
  }
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < m; j++) {
      cout << mat[i][j];
    }
    cout << endl;
  }
  return 0;
}