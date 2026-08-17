#include <iostream>
#include <fstream>
#include <vector>
#include <random>
#include <cmath>
#include <iomanip>
#include <algorithm>
using namespace std;
const int N = 10000;
const int NUM_BINS = 10;
const double PI = 3.14159265358979323846;
double calculateMean(const vector<double>& data)
{
 double sum = 0.0;
 for (double x : data)
 {
 sum += x;
 }
 return sum / data.size();
}
double calculateVariance(
 const vector<double>& data,
 double mean)
{
 double sum = 0.0;
 for (double x : data)
 {
 sum += (x - mean) * (x - mean);
 }
 return sum / data.size();
}
double standardNormalCDF(double z)
{
 return 0.5 *
 (1.0 + erf(z / sqrt(2.0)));
}
double gaussianCDF(
 double x,
 double mu,
 double sigma)
{
 double z =
 (x - mu) / sigma;
 return standardNormalCDF(z);
}
void displayHistogram(
 const vector<double>& data,
 double minValue,
 double maxValue,
 const string& title)
{
 vector<int> frequency(NUM_BINS, 0);
 double binWidth =
 (maxValue - minValue)
 / NUM_BINS;
 for (double value : data)
 {
 int bin =
 static_cast<int>(
 (value - minValue)
 / binWidth
 );
 if (bin >= 0 &&
 bin < NUM_BINS)
 {
 frequency[bin]++;
 }
 if (value == maxValue)
 {
 frequency[NUM_BINS - 1]++;
 }
 }
 int maxFrequency =
 *max_element(
 frequency.begin(),
 frequency.end()
 );
 cout << "\n\n";
 cout << title << endl;
 cout << "===================================================="
 << endl;
 for (int i = 0;
 i < NUM_BINS;
 i++)
 {
 double lower =
 minValue
 + i * binWidth;
 double upper =
 lower + binWidth;
 int barLength = 0;
 if (maxFrequency > 0)
 {
 barLength =
 static_cast<int>(
 50.0 *
 frequency[i] /
 maxFrequency
 );
 }
 cout << fixed
 << setprecision(2);
 cout << "["
 << setw(7)
 << lower
 << " , "
 << setw(7)
 << upper
 << "] | ";
 for (int j = 0;
 j < barLength;
 j++)
 {
 cout << "*";
 }
 cout << " ("
 << frequency[i]
 << ")"
 << endl;
 }
}
int main()
{
 double a;
 double b;
 double mu;
 double variance;
 cout << "============================================"
 << endl;
 cout << "RANDOM VARIABLE GENERATION"
 << endl;
 cout << "============================================"
 << endl;
 cout << "\nEnter lower limit a for U(a,b): ";
 cin >> a;
 cout << "Enter upper limit b for U(a,b): ";
 cin >> b;
 cout << "Enter Gaussian mean (mu): ";
 cin >> mu;
 cout << "Enter Gaussian variance (sigma^2): ";
 cin >> variance;
 if (b <= a)
 {
 cout << "\nError: b must be greater than a."
 << endl;
 return 1;
 }
 if (variance <= 0)
 {
 cout << "\nError: Variance must be greater than zero."
 << endl;
 return 1;
 }
 double sigma =
 sqrt(variance);
 random_device rd;
 mt19937 generator(rd());
 uniform_real_distribution<double>
 uniformAB(a, b);
 vector<double> uniformData;
 uniformData.reserve(N);
 for (int i = 0;
 i < N;
 i++)
 {
 double u =
 uniformAB(generator);
 uniformData.push_back(u);
 }
 ofstream uniformFile(
 "uniform_data.txt"
 );
 if (!uniformFile)
 {
 cout << "Error creating uniform_data.txt"
 << endl;
 return 1;
 }
 for (double x : uniformData)
 {
 uniformFile << setprecision(15)
 << x
 << endl;
 }
 uniformFile.close();
 uniform_real_distribution<double>
 uniformMinusOneOne(
 -1.0,
 1.0
 );
 vector<double> gaussianData;
 gaussianData.reserve(N);
 while (gaussianData.size() < N)
 {
 double u1 =
 uniformMinusOneOne(
 generator
 );
 double u2 =
 uniformMinusOneOne(
 generator
 );
 double s =
 u1 * u1
 +
 u2 * u2;
 if (s > 0.0 &&
 s < 1.0)
 {
 double k =
 sqrt(
 (-2.0 * log(s))
 / s
 );
 double x =
 u1 * k;
 double y =
 u2 * k;
 double xPrime =
 mu
 +
 sigma * x;
 double yPrime =
 mu
 +
 sigma * y;
 gaussianData.push_back(
 xPrime
 );
 if (gaussianData.size() < N)
 {
 gaussianData.push_back(
 yPrime
 );
 }
 }
 }
 ofstream gaussianFile(
 "gaussian_data.txt"
 );
 if (!gaussianFile)
 {
 cout << "Error creating gaussian_data.txt"
 << endl;
 return 1;
 }
 for (double x : gaussianData)
 {
 gaussianFile << setprecision(15)
 << x
 << endl;
 }
 gaussianFile.close();
 double calculatedMean =
 calculateMean(
 gaussianData
 );
 double calculatedVariance =
 calculateVariance(
 gaussianData,
 calculatedMean
 );
 cout << "\n\n";
 cout << "============================================"
 << endl;
 cout << "GAUSSIAN RANDOM VARIABLE RESULTS"
 << endl;
 cout << "============================================"
 << endl;
 cout << fixed
 << setprecision(6);
 cout << "Number of Samples : "
 << N
 << endl;
 cout << "Required Mean (mu) : "
 << mu
 << endl;
 cout << "Required Variance : "
 << variance
 << endl;
 cout << "Standard Deviation : "
 << sigma
 << endl;
 cout << "Calculated Mean : "
 << calculatedMean
 << endl;
 cout << "Calculated Variance : "
 << calculatedVariance
 << endl;
 displayHistogram(
 uniformData,
 a,
 b,
 "UNIFORM RANDOM VARIABLE HISTOGRAM"
 );
 double gaussianMin =
 mu - 4.0 * sigma;
 double gaussianMax =
 mu + 4.0 * sigma;
 displayHistogram(
 gaussianData,
 gaussianMin,
 gaussianMax,
 "GAUSSIAN RANDOM VARIABLE HISTOGRAM"
 );
 cout << "\n\n";
 cout << "============================================"
 << endl;
 cout << "CHI-SQUARE GOODNESS-OF-FIT TEST"
 << endl;
 cout << "============================================"
 << endl;
 vector<int> observed(
 NUM_BINS,
 0
 );
 vector<double> expected(
 NUM_BINS,
 0.0
 );
 vector<double> boundaries(
 NUM_BINS - 1
 );
 double rangeMin =
 mu - 4.0 * sigma;
 double rangeMax =
 mu + 4.0 * sigma;
 double binWidth =
 (rangeMax - rangeMin)
 / NUM_BINS;
 for (int i = 0;
 i < NUM_BINS - 1;
 i++)
 {
 boundaries[i] =
 rangeMin
 + (i + 1)
 * binWidth;
 }
 for (double value : gaussianData)
 {
 int bin = 0;
 while (bin < NUM_BINS - 1 &&
 value > boundaries[bin])
 {
 bin++;
 }
 observed[bin]++;
 }
 expected[0] =
 N *
 gaussianCDF(
 boundaries[0],
 mu,
 sigma
 );
 for (int i = 1;
 i < NUM_BINS - 1;
 i++)
 {
 double probability =
 gaussianCDF(
 boundaries[i],
 mu,
 sigma
 )
 -
 gaussianCDF(
 boundaries[i - 1],
 mu,
 sigma
 );
 expected[i] =
 N *
 probability;
 }
 expected[NUM_BINS - 1] =
 N *
 (
 1.0
 -
 gaussianCDF(
 boundaries[NUM_BINS - 2],
 mu,
 sigma
 )
 );
 double chiSquare = 0.0;
 for (int i = 0;
 i < NUM_BINS;
 i++)
 {
 if (expected[i] > 0)
 {
 chiSquare +=
 pow(
 observed[i]
 -
 expected[i],
 2
 )
 /
 expected[i];
 }
 }
 cout << "\n";
 cout << left
 << setw(10)
 << "Bin"
 << setw(20)
 << "Observed"
 << setw(20)
 << "Expected"
 << endl;
 cout << "-----------------------------------------------"
 << endl;
 for (int i = 0;
 i < NUM_BINS;
 i++)
 {
 cout << left
 << setw(10)
 << i + 1
 << setw(20)
 << observed[i]
 << setw(20)
 << fixed
 << setprecision(4)
 << expected[i]
 << endl;
 }
 int totalObserved = 0;
 double totalExpected = 0.0;
 for (int i = 0;
 i < NUM_BINS;
 i++)
 {
 totalObserved +=
 observed[i];
 totalExpected +=
 expected[i];
 }
 cout << "\n";
 cout << "Total Observed Frequency = "
 << totalObserved
 << endl;
 cout << "Total Expected Frequency = "
 << totalExpected
 << endl;
 int degreesOfFreedom =
 NUM_BINS - 1;
 double alpha = 0.05;
 double criticalValue = 16.919;
 cout << "\n";
 cout << "Calculated Chi-Square = "
 << chiSquare
 << endl;
 cout << "Degrees of Freedom = "
 << degreesOfFreedom
 << endl;
 cout << "Significance Level (alpha) = "
 << alpha
 << endl;
 cout << "Critical Chi-Square Value = "
 << criticalValue
 << endl;
 cout << "\n";
 if (chiSquare >
 criticalValue)
 {
 cout << "Decision: Reject the Null Hypothesis."
 << endl;
 cout << "The generated data does not sufficiently"
 << endl;
 cout << "follow the expected Gaussian distribution."
 << endl;
 }
 else
 {
 cout << "Decision: Fail to Reject the Null Hypothesis."
 << endl;
 cout << "The generated data is consistent with"
 << endl;
 cout << "the expected Gaussian distribution."
 << endl;
 }
 cout << "\n";
 cout << "============================================"
 << endl;
 cout << "FILES CREATED SUCCESSFULLY"
 << endl;
 cout << "============================================"
 << endl;
 cout << "1. uniform_data.txt"
 << endl;
 cout << "2. gaussian_data.txt"
 << endl;
 return 0;
}
