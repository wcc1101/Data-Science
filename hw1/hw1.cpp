#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include <algorithm>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <list>
#include <map>
#include <sstream>
#include <vector>
using namespace std;

struct TreeNode {
    int count;
    string name;
    TreeNode* parent;
    TreeNode* next;
    map<string, TreeNode*> children;

    TreeNode(string name, TreeNode* parent, int count = 1) : name(name), parent(parent), count(count) {
        next = NULL;
        children.clear();
    }

    bool isRoot() {
        return (parent == NULL);
    }

    void print(int k) {
        for (int i = 0; i < k; i++)
            cout << "--------";
        if (isRoot())
            cout << endl
                 << "Root" << endl;
        else
            cout << name << ": " << count << endl;
        for (map<string, TreeNode*>::iterator it = children.begin(); it != children.end(); it++) {
            cout << "|";
            it->second->print(k + 1);
        }
    }
};

map<vector<string>, int> transactions;
map<string, int> FreqTable;
vector<string> Rank;
map<string, TreeNode*> heads;
int minSup;
float totalCount = 0.0;

ifstream fin;
ofstream fout;

bool Compare(string a, string b) {
    return (FreqTable[a] > FreqTable[b]);
}

void CTree(map<vector<string>, int>& transactions, map<string, TreeNode*>& heads, TreeNode& root, map<string, int>& FreqTable, vector<string>& Rank) {
    // compare and sort by FreqTableuency
    for (map<vector<string>, int>::iterator it = transactions.begin(); it != transactions.end(); it++)
        for (int i = 0; i < it->first.size(); i++)
            FreqTable[it->first[i]] += it->second;
    for (map<string, int>::iterator it = FreqTable.begin(); it != FreqTable.end(); it++)
        if (FreqTable[it->first] >= minSup)
            Rank.push_back(it->first);
    sort(Rank.begin(), Rank.end(), Compare);

    // construct the tree
    for (map<vector<string>, int>::iterator it = transactions.begin(); it != transactions.end(); it++) {
        vector<string> tran = it->first;
        sort(tran.begin(), tran.end(), Compare);
        int count = it->second;
        TreeNode* cur = &root;
        for (int i = 0; i < tran.size(); i++) {
            string name = tran[i];
            if (FreqTable[name] < minSup)
                continue;
            if (cur->children.find(name) != cur->children.end()) {
                cur = cur->children[name];
                cur->count += count;
            } else {
                cur->children[name] = new TreeNode(name, cur, count);
                cur = cur->children[name];
                if (heads.find(name) != heads.end())
                    cur->next = heads[name];
                heads[name] = cur;
            }
        }
    }
}

void Permutation(vector<string>& set, int start, list<string>& suffix, int freq, int length, map<string, int> subFreqtable) {
    // last one
    if (start == set.size() - 1) {
        suffix.push_back(set[start]);
        if (suffix.size() <= length) {
            return;
        }
        int f = freq;
        for (list<string>::iterator it = suffix.begin(); it != suffix.end();) {
            if (subFreqtable[*it] != 0 && subFreqtable[*it] < f) f = subFreqtable[*it];
            fout << *it;
            if (++it != suffix.end())
                fout << ",";
        }
        fout << ":" << fixed << setprecision(4) << f / totalCount << endl;

        suffix.pop_back();
        f = freq;
        if (suffix.size() > length) {
            for (list<string>::iterator it = suffix.begin(); it != suffix.end();) {
                if (subFreqtable[*it] != 0 && subFreqtable[*it] < f) f = subFreqtable[*it];
                fout << *it;
                if (++it != suffix.end())
                    fout << ",";
            }
            fout << ":" << fixed << setprecision(4) << f / totalCount << endl;
        }
        return;
    }

    // recursion
    suffix.push_back(set[start]);
    Permutation(set, start + 1, suffix, freq, length, subFreqtable);
    suffix.pop_back();
    Permutation(set, start + 1, suffix, freq, length, subFreqtable);
}

void FPG(TreeNode& root, string name, int freq, map<string, TreeNode*>& heads, list<string>& suffix) {
    //  generate sub transactions
    map<vector<string>, int> subTrans;
    TreeNode* cur = heads[name];
    vector<string> path;
    // root.print(0);
    do {
        TreeNode* leaf = cur;
        path.clear();
        int count = leaf->count;
        while ((leaf = leaf->parent) != NULL) {
            if (leaf->isRoot())
                break;
            path.push_back(leaf->name);
        }
        if (path.size() > 0)
            subTrans[path] = count;
    } while ((cur = cur->next) != NULL);

    // construct conditional tree
    map<string, TreeNode*> subHeads;
    TreeNode subRoot("subRoot", NULL);
    map<string, int> subFreqtable;
    vector<string> subRank;
    CTree(subTrans, subHeads, subRoot, subFreqtable, subRank);
    // subRoot.print(0);

    // mutiple path
    cur = &subRoot;
    int single = 1;
    do {
        if (cur->children.size() > 1) {
            single = 0;
            break;
        } else if (cur->children.size() <= 0)
            break;
        cur = cur->children.begin()->second;
    } while (cur);
    int f = freq;
    suffix.push_back(name);
    for (list<string>::iterator it = suffix.begin(); it != suffix.end();) {
        if (subFreqtable[*it] != 0 && subFreqtable[*it] < f) f = subFreqtable[*it];
        fout << *it;
        if (++it != suffix.end())
            fout << ",";
    }
    fout << ":" << fixed << setprecision(4) << f / totalCount << endl;

    if (single) {
        // one path
        if (subRank.size() > 0)
            Permutation(subRank, 0, suffix, subFreqtable[subRank[0]], suffix.size(), subFreqtable);
        suffix.pop_back();
    } else {
        // mutiple path
        for (int i = subRank.size() - 1; i >= 0; i--)
            FPG(subRoot, subRank[i], subFreqtable[subRank[i]], subHeads, suffix);
    }
}

int main(int argc, char* argv[]) {
    // input
    fin.open(argv[2]);
    fout.open(argv[3]);
    string str, tok;
    vector<string> tran;
    while (fin >> str) {
        tran.clear();
        istringstream ss(str);
        while (getline(ss, tok, ','))
            tran.push_back(tok);
        transactions[tran]++;
        totalCount++;
    }
    float x = totalCount * atof(argv[1]);
    if (x > (int)x)
        minSup = (int)x + 1;
    else
        minSup = (int)x;
    // cout << minSup << endl;

    // construct the conditional tree
    TreeNode root("root", NULL);
    CTree(transactions, heads, root, FreqTable, Rank);
    // root.print(0);

    list<string> suffix;
    // mine the FP
    for (int i = Rank.size() - 1; i >= 0; i--) {
        string name = Rank[i];
        suffix.clear();
        FPG(root, name, FreqTable[name], heads, suffix);
    }
    // cout << (double)clock() / CLOCKS_PER_SEC << "s";
    return 0;
}
