## 선형 회귀 Linear Regression

**머신 러닝의 가장 큰 목적은 실제 데이터를 바탕으로 모델을 생성해서 입력 값을 넣었을때의 결과를 예측하는 데에 있습니다.**

저희가 찾아낼 수 있는 가장 직관적이고 간단한 모델은 선입니다.

그래서 데이터를 놓고 그걸 가장 잘 설명할 수 있는 선을 찾는 분석하는 방법을 선형 회귀(Linear Regression) 분석이라 합니다.

예를 들어 공부 시간과 성적에 데이터를 펼쳐 놓고 그것에 대해 가장 잘 설명할 수 있는 선을 그리면, 공부를 몇 시간 해서 성적이 몇점이 

나올지 예측을 할 수 있습니다.




![img](data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAW4AAAFuCAYAAAChovKPAAAAOXRFWHRTb2Z0d2FyZQBNYXRwbG90bGliIHZlcnNpb24zLjMuMiwgaHR0cHM6Ly9tYXRwbG90bGliLm9yZy8vihELAAAACXBIWXMAAAsTAAALEwEAmpwYAAA+OklEQVR4nO3deXzV1Z3/8dfnbtkDCUnYt7AGNxSkWqpicUHA2umvC3aZqp3STWvHmVathZlBbe1ip0xXrW2n01bpMnXGIiIuVastAlq1mrAGMIEAIft298/vj3txUprk3iT35i75PB8PHsm9+Z7vPReSN9+c7zmfI6qKMcaYzOFIdQeMMcYMjgW3McZkGAtuY4zJMBbcxhiTYSy4jTEmw7hS3YFEWrFihW7dujXV3TDGmKGSeA7KqivukydPproLxhiTdFkV3MYYMxpYcBtjTIax4DbGmAxjwW2MMRnGgtsYYzKMBbcxxmQYC25jjMkwFtzGGJNhLLiNMSbDWHAbY0yGseA2xpgMY8FtjDEZxoLbGGMyjAW3McakAV8wFPexFtzGGJNi3kCIhlZv3Mdn1UYKxhiTaXr8IY63ewmrxt3GgtsYY1KkyxfkRIcPHURogwW3McakRIc3wMlO/6BDGyy4jTFmxLV7A5zs8A25fdKCW0TmAb/s9VQlsB6YDFwN+IEDwPWq2tpH+0NABxACgqq6OFl9NcaYkdLa7ae5yz+scyRtVomq7lHVhaq6EFgEdAMPA08AZ6rq2cBe4PYBTnNp9BwW2saYjNfcNfzQhpGbDrgcOKCqh1V1m6oGo89vB6aMUB+MMSZlTnb6aO0efmjDyAX3GuChPp6/AXisnzYKbBORl0RkbX8nFpG1IrJLRHY1NjYmoKvGGJM4qsqJdi/tPYGEnTPpwS0iHuBdwK9Pe/4OIAj8op+mS1X1POAq4DMicnFfB6nq/aq6WFUXl5eXJ7DnxhgzPOGwcqzdS6cvGPvgQRiJK+6rgJdV9fipJ0Tko8Bq4EPaz1wYVT0a/XiCyNj4khHoqzHGJEQorDS0e+nxx7+UPV4jEdzX0muYRERWALcC71LV7r4aiEiBiBSd+hy4Anh9BPpqjDHDFgiFOdragy+Q+NCGJAe3iOQDlwO/7fX0d4Ai4AkReUVEfhA9dpKIbIkeMx54XkReBXYAj6rq1mT21RhjEsEXjNQdCYTCSXuNpC7AiV5Rjzvtudn9HHsUWBn9vBY4J5l9M8aYRPMGInVHQuHBr4YcDFs5aYwxCdDtD3K8ffB1R4bCgtsYY4ap0xekcQjFoobKgtsYY4ZhuHVHhsKC2xhjhigRdUeGwoLbGGOGoLnLn7Al7INlwW2MMYN0stOX0CXsg2XBbYwxcVJVGjt9dHoTu4R9sCy4jTEmDqrK8XYf3f7UhjZYcBtjTEzhsHK8Izl1R4bCgtsYYwYQilb4S1bdkaGw4DbGmH4EQ2Ea2pJbd2QoLLiNMaYPgVCYY2kY2mDBbYwxf8MfjIR2MJx+oQ0W3MYY81dGqsLfcFhwG2NMlDcQ4libl/AIFYsaKgtuY4xhZMuyDpcFtzFm1BvpsqzDZcFtjBnVUlGWdbgsuI0xo1Zbd4CmrswKbbDgNsaMUqksyzpcFtzGmFEn1WVZh8uC2xgzqpzo8Ka8LOtwWXAbY0YFVeVEh48uX2aHNlhwG2NGgXQryzpcFtzGmKyWjmVZh8uC2xiTtUJhpaGtB38wPYtFDZUFtzEmK6VrLe1EcCTrxCIyT0Re6fWnXUQ+JyKlIvKEiOyLfizpp/0KEdkjIvtF5LZk9dMYk338wTBHWzMrtA80dsZ9bNKCW1X3qOpCVV0ILAK6gYeB24CnVHUO8FT08V8RESfwXeAqYAFwrYgsSFZfjTHZwxcM0dDWk7a1tE+nqvzu1aN8+hcvx91mpIZKlgMHVPWwiFwDLIs+/1PgGeDW045fAuxX1VoAEdkEXANUj0hvjTEZKVPKsp7S5Qty77a9PLO3cVDtRiq41wAPRT8fr6oNAKraICIVfRw/Gajr9bgeeFtfJxaRtcBagGnTpiWsw8aYzJJJZVkB9h7vYMPmao62egG4ZG553G2THtwi4gHeBdw+mGZ9PNfnv4aq3g/cD7B48eLM+BczxiRUly/IiQwpy6qqPPzno9z33AECIcXtFD5z6WyuPnti3OcYiSvuq4CXVfV49PFxEZkYvdqeCJzoo009MLXX4ynA0ST30xiTgTq8ARozpCxrhzfA1x/fy/P7TwIwpSSPf1m9gFkVhYM6z0gE97X83zAJwCPAR4F7oh//t482O4E5IjITOEJkqOWDSe6nMSbDZFJZ1uqj7dz5aDXH2yP9vayqgn+8bC55Huegz5XU4BaRfOBy4BO9nr4H+JWIfAx4E3hf9NhJwAOqulJVgyJyI/A44AR+rKpvJLOvxpjM0tLlpyUDyrKGVfn1rnoeeP4gobCS43Lw2eVzWHHGeET6GhWOLanBrardwLjTnmsiMsvk9GOPAit7Pd4CbElm/4wxmamp00dbBpRlbesOcM/W3bx4sBmAGePyWX/1AmaMKxjWeW3lpDEmo2RKWdZX61u5+9EaTnZGfitYeeYEbnznbHLdgx8aOZ0FtzEmI2RKWdZQWHlox5v85x8PEVbIczu55fI5LK8an7DXsOA2xqQ91UiFv3Qvy9rc5ecrW2p46c1WAGaXF7JudRVTS/MT+joW3MaYtBaOlmX1pnlZ1pcPt3D3lhpauiNj79ecM4lPLZuFx5X4yiIW3MaYtJUJZVlDYeW//nSIn29/EwUKPE7++cp5g1oJOVgW3MaYtJQJZVkbO3zcvaWG1+rbAJg3oYh1q6qYNDYvqa9rwW2MSTuBUJhjaR7aLx5s4p7H9rw1LfG9iybz8YsqcTsHPzSyo7aZTTvreKW+tfbQPasqYx1vwW2MSSu+YIjjbb60LcsaDIX58QuH2LQzUgevKNfFF66cx9LZZUM6347aZjY+vQ+XQwCa42ljwW2MSRveQIjj7V5C4fQsFnWs3ctdm2uobmgH4IxJxXxpVRXji3OHfM5NO+twOYS8QczvtuA2xqSFHn8ktNO1lvYL+0/ytcf30BFd/LPm/KncsHQGriEMjfTW0N7DmDz3oIZYLLiNMSmXzmVZA6Ew9z1Xy29fPgLA2Dw3t6+cz/kzSod9brfTwfTSApq6fOS44q9bkrSty4wxJh4d3kDahvbR1h4++9Arb4X2wqljuP/vFw07tB0ilBZ4mFKSx6eXzSIQUrr98a8ItStuY0zKtPUEaOpMz7Ksz+xp5N5te+jyhxDgIxdO5yMXTMfpGFpFv1OKct2U5LvfGmJZNr+CDcB9z9VyoLErrv8RLLiNMSnR2u2nuSv9yrL6g2G+98wBHnk1sndLaYGHL66cz3nTSoZ13ly3k3GFHnJcf3sTctn8CpbNrwCIORUQLLiNMSnQ3OWnNQ1radc1d7NhczUHGrsAWDS9hNuvmk9pgWfI53Q7HZQWeCjISVzcWnAbY0ZUY4ePDm/61dJ+ovo4//7kXryBMA6BG5bOZM2SqTiGuNmBQ4Sx+W7G5LmHvGFCfyy4jTEjQlVp7PDRmWZlWXsCIb791H62vnEMgLJCD19aVcXZU8YO+ZxFuW5KCzzDHg/vjwW3MSbpVJXj7b5BzZwYCQdPdrFhczWHm7oBuKCylFuvnM+YfPeQzpfncVJa0Pc4diJZcBtjkiocVo53pFctbVXlsdeP8e2n9+MLhnE6hI9fNJP3LZoypGGNZIxjD8SC2xiTNKFoLW1fGtXS7vYH+daT+3iy5gQAE4pzWbe6iqqJxYM+l0OEknwPxXmuhI9jD8SC2xiTFMFQmGPt3rSqpb3/RCcbNldT39IDwDtml/GFK+dRmDv4KEz2OPZALLiNMQmXbmVZVZVHXm3ge8/sJxBS3E7hU5fM4pqFkwZ9pTxS49gDseA2xiSUPxgJ7XQpy9rpDfKNJ/bw3N6TAEwem8f61VXMGV80qPOM9Dj2QFLfA2NM1vAFQxxrS5+yrLuPtXPn5hoa2rwAXDqvnFsunzuo8E3VOPZALLiNMQnhDURCOx3Ksqoqv3n5CD98rpZgWPG4HNx06WxWnjVhUOFbnOemJD8149gDseA2xgxbtz/I8fb0qPDX3hPgq1v38KfaJgCmleazfnUVleWFcZ8jz+NkXEFOUnZoTwQLbmPMsHT6gjSmSVnW14+0cdejNZzoiFQcvPKM8Xx2+Zy4d5dxOoSywpy0GMceSFJ7JyJjgQeAMwEFbgA+B8yLHjIWaFXVhX20PQR0ACEgqKqLk9lXY8zgpUtZ1rAqv9xZx4+eP0hYIdfl4HOXzeGKMybEfY48j5Pywpxh72gzEpL938pGYKuqvldEPEC+qn7g1BdF5F6gbYD2l6rqyST30RgzBC1dflrSoMJfa7efex7bzY5DLQBUlhWwfvUCpo3Lj6u9iFCa7xnyMvdUSFpwi0gxcDFwHYCq+gF/r68L8H7gncnqgzEmOU52+mjvSX2Fv1fqWrn70RqaonW9rz57Ip9eNoucOIdG3E4HFcU5KZ2TPRTJvOKuBBqBn4jIOcBLwM2q2hX9+kXAcVXd1097BbaJiAL3qer9fR0kImuBtQDTpk1LZP+NMadJlwp/obDy8+2H+dn2w4QV8j1Obrl8Lu+MbEYQl6JcN2WFnrSZ4jcYyRzMcQHnAd9X1XOBLuC2Xl+/FnhogPZLVfU84CrgMyJycV8Hqer9qrpYVReXl5cnqOvGmNOFo3VHUh3aTZ0+Pv+b1/jpnyKhPaeikPs+vCju0HY5HEwYk0t5UU5GhjYk94q7HqhX1Rejj39DNLhFxAW8B1jUX2NVPRr9eEJEHgaWAM8lsb/GmH6kS7GonYea+cqW3bRGh2nevXASn7xkVtzT9gpzXIwrzEm7edmDlbTgVtVjIlInIvNUdQ+wHKiOfvkyYLeq1vfVVkQKAIeqdkQ/vwLYkKy+GmP6lw51R0Jh5T//eIgHX3wTJRLAn79yHhfNKYurvdMhjCvMoTDNp/nFK9nv4ibgF9EZJbXA9dHn13DaMImITAIeUNWVwHjg4eivMS7gQVXdmuS+GmNO4wuGON7mS2ndkcYOH3c9Ws1fjrQDMH9CEetXL2DCmNy42ud7XJQVejJiml+8JB0mzSfK4sWLddeuXanuhjFZIR2WsG+vbeKex3bT7o2Mq79/8RQ+9o6ZuJ0OdtQ2s2lnHQ3tPUwszmPN+VNZUln6VluHCKWFHopzM2eaHxDXGE52/N5gjEmoVC9hD4TC/Oj5g/xqV2Q0tTjXxW1XzeeCynEA7KhtZuPT+3A5hOJcF01dPjY+vY+bmcOSylJy3U7Ki3JwZ9FVdm8W3MaYv9LhDXCy05+y0D7W5uXOR6upaegA4KzJxXxp1QLKi3LeOmbTzjpcDnlrKXue20lPIMSmXXWsOGsCY/M9Ken7SLHgNsa8pa07QFNX6paw/2HfSb7++B46fUEE+ODbpnHd22f8zSyQhvYeik/btSbP7eRkhzfrQxssuI0xUU2dPtpStBrSHwzzg2cP8D+vHAWgJN/N7VfNZ/GM0j6Pn1icR1OX760rbodD8AdDTC0tGLE+p1J2DgAZY+Kmqpzo8KYstI+09HDjQ39+K7TPnTaW+z+yqN/QBlhz/lSCYcUbDOF0CIFQmGAYPnFx5Uh1O6XsituYUUxVOd7uo9ufmtWQT+8+wTef2Eu3P4RD4O8vnM6H3jY95gKZJZWlfN41j1/tquNIaw9TSvL5xMWVLBvEkvdMZsFtzCgVCivH2714U7Aa0hcI8d1nDrD5tQYAxhV4uGNVFQunjo3ZVkQoyXfznkVTeM+iKUnuaXqy4DZmFAqGwjSkaDXk4aYuNmyu4eDJSL25JTNKuO2q+XHdVHQ7HZQX5ZAbZ/W/bGXBbcwok8pd2B9/4xgbn9yHNxjGIfAP75jJ+8+fiiOOYk+FOS7KCnNwZHidkUSw4DZmFPEGQhxvH/ld2Hv8ITY+tY9t1ccBqCjK4Uurqjhz8piYbUWEcZm3AjKpLLiNGSVStTdkbWMnGzbX8GZzNwBvnzWOL1w5j+K82EGcqRsdJJsFtzGjQCoW1qgqj/6lge/8/gD+YBiXQ/jEJZW859zJcdXBzuSNDpLNgtuYLJeKbca6fEG++cRefr+nEYCJY3JZt7qK+ROKY7Z1iFBWlD0lWJPB/maMyVKRhTU+unrtWBOrol4i7D3ewZ2bazjS2gPAxXPL+Ocr5sUVxB6Xg4qi3Lg3RhitLLiNyUKnthnrPUc7VkW94VJV/ueVo/zg2QMEQorbKXx62Wzedc5EGxpJMAtuY7JMKKw0tPXgD/71dL9+K+rtrBt2cHd6g3x92x7+sO8kAFNK8li/egGzKwpjtnU6hLLCHApsaCRu9jdlTBYZaGFNXxX1ct0OjrX3DOs1axrauXNzDcfavQBcVlXB5y6bQ74ndrzkeZyUF+Zk1e40I8GC25gsEWthzekV9QC8gTATivOG9HphVX7zUj0//MNBQmElx+XgpnfO5qozJ8Q13FGS76GkIPtLsCaDBbcxWcAXjGwzNtDCmjXnT2Xj0/voCYTIdTvwBsIEw8qa86cO+vXaugN89fHdbK9tBmD6uHzWr17AzLLYZVVdjsiy9TyPzc0eKgtuYzJcvKshl1SWcjNz2LSzjmPtPUwY4qyS1+pbuevRGk52+gG46swJ3PTO2XHVD8nzOKkoyo1Z/c8MzILbmAzW44+Edrwb+i6pLB3yjciwKg/teJOfvHCIsEbGx2+5fC6XVY2Pq70NjSSOBbcxGWokl7A3d/n5ymO7eelwCwCzygtYv3oBU0vzY7a1oZHEs+A2JgON5BL2l99s4ctbdtPcFRkaedc5k/j0sllxLZLJ97goL8qxoZEEs+A2JoOoKo2dPjq9yd+xJhRWfvanw/xs+2EUKPA4+acr5rFsXnnMtiJCab6HMflW0S8ZLLiNyRAjuWNNY4ePL2+p4dX6NgDmji9k3eoFTB4be+qgVfRLPgtuYzJAjz9EY4dvRDY/2HGwma88tvutzYP/33mT+fhFlXENjRTnuRlXYMvWk82C25g019zlp7Xbn/TXCYbC/PiFQ2zaWQdAUa6LL1w5j6Wzy2K2dTqE8qKcuFZLmuGzv2Vj0tRI7sB+vN3LnZtrqG5oB2DBxGLWra5ifHFuzLZ2A3LkJTW4RWQs8ABwJqDADcCVwMeBxuhhX1TVLX20XQFsBJzAA6p6TzL7akw66au6X7K8sP8kX3t8Dx3RG55rzp/KDUtnxKwfYjcgUyfZV9wbga2q+l4R8QD5RIL731X1G/01EhEn8F3gcqAe2Ckij6hqdZL7a0zK9VfdL9ECoTD3P1fLf798BIAxeW5uv2o+S2bGXqBjNyBTK2nBLSLFwMXAdQCq6gf8cd60WALsV9Xa6Lk2AdcAFtwmqwVCkUJRfVX3S6SjrT3cubmGPcc7ADhnyhjuWFVFWWFOzLaFuS7KCmy39VRK5hV3JZHhkJ+IyDnAS8DN0a/dKCJ/D+wC/klVW05rOxmo6/W4HnhbXy8iImuBtQDTpk1LXO+NGWGxqvslyrN7G/nG43vo8ocQ4CMXTOcjF06POUbtiO62XmS7radcMovguoDzgO+r6rlAF3Ab8H1gFrAQaADu7aNtX99Bfa7rVdX7VXWxqi4uL4+9MMCYdOQNhGho60lqaPuDYb715D7+7XfVdPlDlBZ4+Pr7zua6pTNihrbH5WDS2DwL7TSRzCvueqBeVV+MPv4NcJuqHj91gIj8ENjcT9vetSanAEeT1VFjUmmwhaKGoq65mw2bqznQ2AXAoukl3H7VfErjKPpkc7PTT9KCW1WPiUidiMxT1T3AcqBaRCaqakP0sL8DXu+j+U5gjojMBI4Aa4APJquvxqRKly/IiSQXinqy5jjffGIv3kAYh8D1S2dw7ZJpOGIEsW0plr6S/S9yE/CL6IySWuB64D9EZCGRoY9DwCcARGQSkWl/K1U1KCI3Ao8TmQ74Y1V9I8l9NWZEtXsDnOxIXqEobyDEd57ez5bXjwFQVujhS6uqOHvK2Jhtc91OKopsS7F0JSNREnKkLF68WHft2pXqbhgTU7Kr+x082cWdm6s51NQNwAWVpdx65fy45lyPzfdQku+2oZHUiOsv3X4HMmaEJXMJu6qy9fVj/MfT+/EFwzgdwj+8YybvWzwl5tCI1c3OHBbcxoygxg4fHd5AUs7d7Q/yrSf38WTNCQDGF+ewbtUCFkwqjtnWlq1nFgtuY0aAqtLY4aPTl5y6I/tPdLJhczX1LT0ALJ09ji9cOS/m9D1btp6ZLLiNSbJwWDne4aXHn/i6I6rKI6828L1n9hMIKW6n8ImLK/m7cyfHHKO2ZeuZy4LbmCQKRYtF+ZJQLKrTF+TebXt5dm+kXtuksbmsX72AueOLYrYtynVTVmhzszOVBbcxSRIMhWlIUt2RPcc62LC5moY2LwCXzivnlsvnxpxz7RChrCiHQpubndHsX8+YJEhWsShV5bd/PsJ9z9YSDCsel4MbL53FqrMmxrx6zonOzXbb3OyMZ8FtTIL5giGOtXkJhRO7RqK9J8DXH9/DCweaAJhaksf6qxcwq7wwZlubm51dLLiNSaBk1R1542gbd26u4UR0peUVC8Zz8/I5Medc29zs7GTBbUyCJKPuSFiVX+2s44HnDxJWyHU5uPmyOVx5xoSYbW1udvay4DYmAVq7/TR3JXY1ZGu3n3u27mHHwWYAZpYVsG51FTPGFQzYzuZmZ7+4g1tE8oBp0Up/xhgic7QbO310JXhhzav1rdz1aA1NnZH/DFafPZHPLJtFjnvgIQ+3MzI0khvjOJPZ4gpuEbka+AbgAWZGq/ttUNV3JbFvxqQ1byBEY4cvoTNHQmHlwR1v8tM/HiKskOd2csvlc1leVRGzbWGOi7JC21JsNIj3ivtfiewD+QyAqr4iIjOS0yVj0lsorDR1+ej0JvYqu7nLz5e31PDym60AzK4oZP3qKqaU5A/YzrYUG33iDe6gqrbZVCIz2nX7g5xo9yV81siuQ8185bHdtHRHClC9e+EkPnnJLDyugedce1wOKopyYx5nsku8wf26iHwQcIrIHOCzwB+T1y1j0k8yamiHwsp//vEQD774JgoU5Dj5/JXzuHhO7P1TbUux0Sve4L4JuAPwAQ8S2ZnmrmR1yph0c7LTR3tPYsuxNnb4uOvRav5ypB2A+ROKWLe6iolj8gZsZ1uKmZj/8iLiBB5R1cuIhLcxo0YorJxIQmW/7bVN3PPYbtqj4+TvWzSFf7hoZszl6LZs3UAcwa2qIRHpFpExqto2Ep0yJh14AyFOtPsIhhM3ayQQCvOj5w/yq131ABTnurh1xXwunDUuZtsxeW5KbWjEEP9QiRf4i4g8AXSdelJVP5uUXhmTQqpKS3eAtp5AQldBHmvzcuej1dQ0dABw5qRivrSqiori3AHbOUQoL7KhEfN/4v1OeDT6x5is1ukL0tzp/6ur7B21zWzaWUdDew8Ti/NYc/5UllSWDuq8f9h3kq8/vodOXxABPvi2aVz39hkxl6Pb0IjpS1zBrao/FREPMDf61B5VTc7GecakQCgc2Vqs2//Xc7N31Daz8el9uBxCca6Lpi4fG5/ex83MiSu8/cEw9z1Xy8N/PgJASb6b266az/kzYre1zQ5Mf+JdObkM+ClwiMj28VNF5KOq+lzSembMCOn2B2ns8PVZhnXTzjpcDiEvuoQ8z+2kJxBi0866mMF9pKWHDZur2XeiE4CFU8dyx8r5jCvMGbCdbXZgYon3O+Ne4IpTdUpEZC7wELAoWR0zJtlUleYuP20DTPNraO+hOPevf0xy3Q6OtfcMeO7f7z7BvU/spdsfwiHwkQum8+ELpsccGnE7HYwvtgU1ZmDxBre7d3EpVd0rIra+1mSsYCjMiQ4f3hh7QU4szqOpy/fWFTeANxBmQnHfc619gRDffeYAm19rAGBcgYc7VlWxcOrYmH0qzHVRVmC1Rkxs8Qb3LhH5EfCz6OMPAS8lp0vGJNdAQyOnW3P+VDY+vY+eQIhctwNvIEwwrKw5f+rfHPtmUzcbNldTezIy8WrJjBJuvWo+JfmeAV9DorVGiq3WiIlTvMH9KeAzRJa6C/Ac8L1kdcqYZPAFQ7R0Bf7mBuRAllSWcjNz2LSzjmPtPUzoZ1bJtjeO8a0n9+ENhnEIfOwdM/nA+VNxxLix6HY6qCjOIcdlZVhN/CSeeaoiUgB4VTUUfewEclS1O0a7scADwJmAAjcA7wGuBvzAAeB6VW3to+0hoAMIESlytThWPxcvXqy7du2K+X7M6BIKR8ayO7yJnwjVEwjxH0/t4/E3jgNQUZTDl1ZVcebkMTHbFuS4KLcyrOavxfXNEO8dkKeA3oN6ecCTcbTbCGxV1fnAOUAN8ARwpqqeDewFbh+g/aWqujCe0DbmdKpKa7efuubupIR2bWMnn/75y2+F9oWV47jvI4tihraIMK4gh/HFuRbaZkjiHSrJVdXOUw9UtVNEBiwSLCLFwMXAddE2fiJX2dt6HbYdeO9gOmxMPDq8AVq6Agldrn6KqvLoX47xnd/vxx8M43IIay+u5P+dNznmnGuXIzI0YjvUmOGIN7i7ROQ8VX0ZQEQWAwPPh4JKoBH4iYicQ+Rm5s2q2tXrmBuAX/bTXoFtIqLAfap6f18HichaYC3AtGnT4nw7Jlt5AyGauvz4YswWGaouX5BvPrGX3+9pBGDimFzWra5i/oTimG3zPE4qinJt814zbPGOcZ8PbAKOEgnUScAHVLXfmSXRcN8OLFXVF0VkI9CuquuiX78DWAy8R/vohIhMUtWjIlJBZHjlplgLfmyMe/TyB8O0dPsTvvdjb3uPd3Dn5hqOtEauWS6eW8Y/Xz6PwtzY1z8l+R5KCgaeXWIMcY5xD/gdFw3sOlXdKSLzgU8Qubm4FTgY49z1QL2qvhh9/Bvgtuh5PwqsBpb3FdoAqno0+vGEiDxMZOs0W6lp/koorLR0++nwBhNaEKo3VeV/XjnKD549QCCkuJ3Cp5fN5l3nTIw5NOJ0RApE5XtsFaRJnFg3J+8jMi4NcCHwReC7QAvQ59DFKap6DKgTkXnRp5YD1SKyArgVeFd/s1JEpEBEik59DlwBvB777ZjRoveNx/YEV/HrrdMb5F9/V823n95PIKRMKcnjO9eeyzULJ8UM7Vy3k8lj8yy0TcLF+o5yqmpz9PMPAPer6n8D/y0ir8Rx/puAX0QLVNUC1wM7gRzgieg3/nZV/aSITAIeUNWVwHjg4ejXXcCDqrp1cG/NZKtOX5CWLn9Cd1fvS01DO3duruFYuxeA5fMr+MfL58QVxFY72yRTzOAWEZeqBolcMa8dRFtU9RUi49i9ze7n2KPAyujntUSmDxrzlmTfeDxFVfn1S/X88A8HCYWVHJeDm945m6vOnBAziK12thkJsb67HgKeFZGTRGaR/AFARGYDthuOGRGBUJiWLj+dSbzxeEpbT4Cvbt3N9trIL5rTx+WzfvUCZpYVxGxrtbPNSBkwuFX1bhF5CpgIbOt1I9FBZBjEmKQJR288tifxxmNvf6lv465Ha2jsjOzkvuKMCdy0fPZfFZjqj+24bkZSPMMd2/t4bm9yumNMZKiivSdIa48/rkJQwxVWZdOOOn78wkHCGinb+o+XzeXyBeNjtrXa2SYV7LvNpJVuf5CmzuTfeDylucvPPY/tZtfhFgAqywtYv3oB00oHXBgMgMcVqZ1tQyNmpFlwm7QQDIVp6kruAprT/fnNFu7espvmrsiM16vPmcinL5lFThxDI4W5kQJRNjRiUsGC26RcW3eAlm4/4REYx4bIop2fbT/Mz/50GAUKPE7+6Yq5LJtXEbOt1c426cCC26RMIBTmZKePHn9yp/f1drLTx5e31PBKXWRS1NzxhaxbvYDJY/ve0aY3t9NBeZEViDKpZ8FtUqKtJ0BL18hdZQPsPNTMV7bspjW6x+R7zpvM2osq49rfMd/jorwoxwpEmbRgwW1G1EjffITI+PlP/niIh3bUAVCY4+LWFfNYOrssrvalBR7Gxth+zJiRZMFtRsRQtg1LhBPtXu58tIY3jrYDsGBiEV9avYAJxbkx21rtbJOuLLhNUiVz27BY/njgJF/buod2b+Q/iw8snsLH3jETVxzT9/I8TsoLc+I61piRZsFtkkJVaesJ0NodGNFxbIjc9PzhH2r5zUtHgEjBp9uumsfbZo6Lq/3YfA+lVjvbpDELbpNwI1W9ry8NbT1s2FzDnmMdAJw9ZQx3rKyivCgnZlurnW0yhX2HmoQ5vXrfjtpmNu2so6G9h4nFeaw5fypLKkuT9vrP7m3kG9v20OULIcCHLpjGRy+cEddMkBy3k/FFNjRiMoMFtxm2vqr37ahtZuPT+3A5hOJcF01dPjY+vY+bmZPw8PYHw3z/mQP876tHASjJd/PFlVUsml4SV3srEGUyjQW3GbJwWGntCdDWxw40m3bW4XLIW5X18txOegIhNu2sS2hw17d0s+F3Nexv7ATgvGlj+eLKqrjGqK1AlMlU9h1rhqTdG1lA01/1vob2HopP20Q31+3gWHtPwvrwVM1xvvnEPnoCIRwC1719BtcumRbX0IjH5aCiKDeuxTfGpBsLbjMo8S6gmVicR1OX769qWXsDYSYUx15aHos3EOI7T+9ny+vHABhX6OFLq6o4Z8rYuNpbgSiT6Sy4TVz8wTDNXf64F9CsOX8qG5+OXA3nuh14A2GCYWXN+VOH1Y9DTV1s+F01h5oi+0wvmVnK7SvmMyY/dtEnKxBlsoUFtxnQUBfQLKks5WbmsGlnHcfae5gwzFklqsrWN47zH0/twxcM43QIH3vHTN6/eAqOOK6c3c7IKsgcl62CNJnPgtv0KRELaJZUlibkRmSPP8S3ntrHE9XHAagoymHd6irOmDQmrvaFOS7KCnNwWIEokyUsuM3fSOUCmtMdONHJv22upr4lclNz6axxfGHFPIriGO4QEUrzPXENoxiTSSy4zVu8gRDNXX68gZGrj90fVWXzaw185/f7CYQUl0P45CWV/N25k+O6qWgFokw2s+A2BENhmrv9dHpHtnJffzp9Qb65bS/P7G0EYOKYXP7l6gXMHV8UV3urnW2ynQX3KJbKQlD92Xu8g3/7XTUNbV4Als0t55Yr5sa9SKYk30OJFYgyWc6Ce5Tq8gVpTpNxbIj8J/Lwn4/wg2drCYYVt1O46Z2zWXXWxLiGRpwOoaIolzyPDY2Y7GfBPcr4gpFx7JHc5zGW9p4AX398Dy8caAJgakke669ewKzywrja57qdVFiBKDOKJDW4RWQs8ABwJqDADcAe4JfADOAQ8H5Vbemj7QpgI+AEHlDVe5LZ12yXyg0NBvLG0Tbu3FzDiQ4fAJcvGM/nls+J+8p5bL6Hkny3rYI0o0qyr7g3AltV9b0i4gHygS8CT6nqPSJyG3AbcGvvRiLiBL4LXA7UAztF5BFVrU5yf7NOOo5jA4RV+dXOOh54/iBhhVyXg88un8OKMyfE1d5qZ5vRLGnf9SJSDFwMXAegqn7ALyLXAMuih/0UeIbTghtYAuxX1drouTYB1wAW3IPQ4Q3Q0hUgGE6PcexT2roD3LN1Ny8ebAZgZlkB61ZXMWNcQVztc6JDI24bGjGjVDIvVyqBRuAnInIO8BJwMzBeVRsAVLVBRCr6aDsZqOv1uB54W18vIiJrgbUA06ZNS1zvM1iPP0RTlw9/ML0CG+DV+lbuerSGpk4/ACvPmsCNl86Oe7611c42JrnB7QLOA25S1RdFZCORYZF49PVT2efv+ap6P3A/wOLFi9NnLCAFUrWTejxCYeXBHW/y0z8eIqyR+ty3XD6H5VXj42pvtbON+T/J/CmoB+pV9cXo498QCe7jIjIxerU9ETjRT9veZeSmAEeT2NeMFgyFaekOpN2Nx1Oau/x8eUsNL7/ZCsDs8kLWX13FlJL8uNq7nQ7GF1vtbGNOSVpwq+oxEakTkXmqugdYTmSMuhr4KHBP9OP/9tF8JzBHRGYCR4A1wAeT1ddMFQ5Hbjy29aTXjcfeXj7cwt1bamjpjvyn8u6Fk/jkJbPiDuHCXBdlBVYgypjekv17503AL6IzSmqB6wEH8CsR+RjwJvA+ABGZRGTa30pVDYrIjcDjRKYD/lhV30hyXzNKuzdAaxreeDwlFFb+60+H+Pn2N1GgIMfJ56+Yx8Vzy+Nqb7WzjemfnL5XYCZbvHix7tq1K9XdSKp4d6BJpcYOH3dvqeG1+jYA5k0oYv3qKiaOiW/3G6udbUaxuH61tDs9GSIdVzz25cWDTXxly27aowWr3rdoCv9w0cy4p+4V5ES2FbOhEWP6Z8Gd5tKtcl9/gqEwP3r+IL/cVQ9Aca6LL6yYx9tnlcXV3mpnGxM/C+40FQ4rrdEbj+k+nHWs3ctdm6upbugA4IxJxaxbVUVFcW5c7a12tjGDY8GdZlSVdm+Q1m4/oXB6BzbA8/tO8rXH99Dpi/xG8MElU7nu7TPiLviU53FSUZRrtbONGQQL7jSSbqVWB+IPhrn/D7X89uUjAIzNc3P7yvmcPyP+PSatdrYxQ2PBnQbSacuweBxp7eHOzdXsPd4JwMKpY/jiyirKCnPiam+1s40ZHgvuFAqEwrR0+d8aZsgEz+w5wTe27aXbH0KAv79wOh++YHrcQx1WO9uY4bPgToFQWGnt9tPuDab9jcdTfIEQ33v2AL97tQGA0gIPd6ycz7nTSuI+x5g8N6VWIMqYYbPgHkGqSntPkJZuf9ouUe/Lm83dbNhcTW1jFwCLp5dw+8r5lOTHNz7tkEjt7AIrEGVMQthP0gjp9AVpyZAbj709UX2cf39yL95AGIfADUtnsmbJVBxxXjV7XJECUVY725jEseBOMm8gRFOXH1+G3Hg8pScQ4ttP7WfrG8cAKC/MYd3qKs6cPCbucxTluikrtKERYxLNgjtJMvHG4ykHT3axYXM1h5u6AbigspRbV8xnTF58qxpFhLJCD0VWIMqYpLDgTrBwWGnJsBuPp6gqj71+jG8/vR9fMIzTIay9aCbvXTQl7qtmKxBlTPJZcCdIpq14PF23P8i3ntzHkzWRfS0mFOeybnUVVROL4z5HYY6LMisQZUzSWXAnQCaUWh3I/hOdbNhcTX1LDwAXzSnj81fMozA3vm8PEaG0wBP3UIoxZngsuIchU0qt9kdVeeTVo3zvmQMEQorbKXzqkllcs3DSoIZGyousQJQxI8mCewjSfY/HeHR6g3xj2x6e23cSgMlj81i3uoq544viPke+x0V5UY4ViDJmhFlwD4Kq0tqd3ns8xqOmoZ07N9dwrN0LwDvnV3DL5XPI98T/7VBa4GFsnAtwjDGJZcEdpw5vgJY03uMxHqrKb14+wg+fqyUYVjwuBzdeOptVZ02Ie2jECkQZk3oW3DFk6gKa07X1BPjq1t1sr20GYHppPuuvXsDMsoK4z2EFooxJDxbc/QiEwjR3+enKwAU0p3v9SBt3PVrDiQ4fACvOmMBNy2eTN4gbilYgypj0YcF9mkys3NefsCqbdtTx4xcOElbIdTv43PI5XHHGhLjPYQWijEk/9tMYdapyX2tPZi6gOV1Lt5+vbNnNrsMtAFSWF7B+1QKmjcuP+xw50aERKxBlTHqx4CaztgyLxyt1rdz9aA1NXX4Arj57Ip9eNoucQQyNFOe5GWdDI8akpVEd3L5giKbOzNkyLJZQWPn59sP8bPthwgr5Hif/fMVcls2riPscDhHKinIotKERY9LWqPzpDIbCNHf76fRm/o3HU5o6fdy9ZTev1LUCMHd8IetWL2Dy2Ly4z2EFoozJDKMquMNhpa0nQGtPIONvPPa281AzX9mym9aeyErO95w3mbUXVeJxxT82ne9xUVFkBaKMyQSjJrjbvQFaM3wBzelCYeUnLxzkwR11QKQ63xeunMc75pQN6jwl+R5KCmwVpDGZIqnBLSKHgA4gBARVdbGI/BKYFz1kLNCqqgvjaTuUPvT4QzR1+fAHsyewAU60e7nr0RpeP9oOwIKJRXxp9QImFOfGfQ6HCBXFOYNa6m6MSb2R+Im9VFVPnnqgqh849bmI3Au0xdt2MPzByAKabn/2jGOf8qcDTXx1627ao2P0a86fyg1LZwxqRaPtBWlM5krZpZZE5pm9H3hnIs8biu5A05EFC2hOFwiFeeAPB/n1S/UAFOe6uO2q+VxQOW5Q57END4zJbMkObgW2iYgC96nq/b2+dhFwXFX3DaHtW0RkLbAWYOrUadQ1d2d05b7+NLT1cOfmGnYf6wDgrMlj+NKqKsqLcuI+h4hQmu9hTL5teGBMJkt2cC9V1aMiUgE8ISK7VfW56NeuBR4aYtu3RAP9foCzFp6n2Rjaz+1r5OuP76HLF0KAD10wjY9eOGNQdbBdjshUP9vwwJjMl9TgVtWj0Y8nRORhYAnwnIi4gPcAiwbbNpn9TTf+YJgfPHuA/3nlKAAl+W6+uLKKRdNLBnWePI+TiqJc2/DAmCyRtOAWkQLAoaod0c+vADZEv3wZsFtV64fQdlSob+lmw+Ya9p/oBODcaWO5Y2UVpYOctjcmz824wviHU4wx6S+ZV9zjgYejtS5cwIOqujX6tTWcNkwiIpOAB1R1ZYy2WW1HbTPff/YAbzZ3o4AA1y2dwQeXTBvUFbOIUFbooSjXxrONyTZJC25VrQXO6edr1/Xx3FFgZay22ez5vSf5ytbd9ERrpzgFxuR7mFdRNKjQtqXrxmQ3m8SbJg43dfHlx2reCu18j5Pp4woo8DjZtLMu7vPkeZxMGptnoW1MFrMlc2ng8TeOsfHJfXijqzvLCjyU5LsREZwOB8fae+I6j+1SY8zoYMGdQj3+EBuf2se26uMAuJ1CSb6bsXn/dwPSGwgzoXjgCn82nm3M6GLBnSIHGjvZ8Ltq6loiV9NLZ41j+fwKfvj8QXoCIXLdDryBMMGwsub8qf2ex8azjRl9LLhHmKqy+bUGvvP7/QRCisshfPKSSv7u3MmICPkeF5t21nGsvYcJxXmsOX8qSypL+zyXzc82ZnSy4B5BXb4g927byzN7GwGYOCaX9asXMG9C0VvHLKks7Teoe7P52caMXhbcI2Tv8Q42bK7maKsXgGVzy7nlirmD3iJMoruu29Zixoxe9tOfZKrKw38+wn3P1RIIKW6ncOOls1l99sRBz/6w8WxjDFhwJ1WHN8DXHt/DC/ubAJhaksf61QuYVVE46HPZeLYx5hQL7iSpPtrOnY9Wc7zdB8BlVRX842VzyfMM/mp5bL5n0DVKjDHZy4I7wcKq/HpXPQ88f5BQWMlxOfjs8jmsOGP8oIdGHCKU2Xi2MeY0lggJ1NYd4J6tu3nxYDMAM8bls/7qBcwYVzDoc9l4tjGmPxbcCfJafSt3PVrDyU4/ACvPnMCN75w9pI0L8j0uyotybDzbGNMnC+5hCqvy0I43+ckLhwgr5Lmd3HL5HJZXjR/S+Ww82xgTiwX3MDR3+fnKY7t56XALALPKC1i/egFTS/MHfS4bzzbGxMtSYohePtzC3VtqaOkOAHDNOZP41LJZeFyDr5TrdjoYX5w7pLbGmNHHgnuQQmHlv/50iJ9vfxMFCjxO/vnKeVwyt3xI58v3uKgoysFh49nGmDhZcA9CY4ePu7fU8Fp9GwDzxhexbnUVk8YOXHa1PzaebYwZCgvuOL14sIl7HttDW09kaOS9iybz8YsqcTsHP7zhiNYbKbDxbGPMEFhyxBAMhfnxC4fe2j6sKNfFF66cx9LZZUM6n41nG2OGy4J7AMfbvdy5uYbqhnYAFkwsZt3qKsYX5w7pfDaebYxJBAvufryw/yRfe3wPHd4gANcumcr1b5+BawhDIwAl+R5KbDzbGJMAFtynCYTC3PdcLb99+QgQ2bDg9qvms2Rm7M0N+mLj2caYRLM06eVIaw93ba5hz/EOAM6ZMoY7VlVRNsSdZmw82xiTDBbcUc/saeTebXvo8ocQ4CMXTOcjF04fcr0QG882xiTLqA9ufzDM9545wCOvHgWgtMDDF1fO57xpJUM+p41nG2OSaVQH95vN3dy5uZoDjV0ALJpewu1XzR/yohiHCBXFOeR7RvVfqzEmyZKaMCJyCOgAQkBQVReLyL8CHwcao4d9UVW39NF2BbARcAIPqOo9iezbE9XH+fcn9+INhHEIXL90BtcumYZjkJsdnGLj2caYkTISl4aXqurJ0577d1X9Rn8NRMQJfBe4HKgHdorII6paPdzO9ARCfPup/Wx94xgAZYUe1q1awFlTxgz5nIU5LsoKbTzbGDMy0vV3+iXAflWtBRCRTcA1wLCC++DJLjZsruZwUzcAF1SWcuuV8xmT7x7yOa3eiDFmpCX793oFtonISyKyttfzN4rIayLyYxHp6y7gZKCu1+P66HN/Q0TWisguEdnV3HT6hX20E6o89pcGPv2Llznc1I3TIXzykkrufveZwwrtsqIcC21jzIhL9hX3UlU9KiIVwBMishv4PnAnkVC/E7gXuOG0dn2NOWhfL6Cq9wP3A5y18Ly/OabbH+RbT+7jyZoTAEwozmXd6iqqJhYP8S3ZTUhjTGolNXlU9Wj04wkReRhYoqrPnfq6iPwQ2NxH03pgaq/HU4Cjg339/Sc62bC5mvqWHgAumlPG56+YR2Hu0N+23YQ0xqRa0oJbRAoAh6p2RD+/AtggIhNVtSF62N8Br/fRfCcwR0RmAkeANcAH431tVeWRVxv43jP7CYQUt1P41CWzuGbhJGSIs0YA8jxOKopybRNfY0xKJfOKezzwcDQoXcCDqrpVRH4mIguJDH0cAj4BICKTiEz7W6mqQRG5EXicyHTAH6vqG/G8aKcvyL3b9vLs3shsw8lj81i/uoo544uG9WbG5LkZN8Sl78YYk0ii2ufQcUaac8Y5Wvbhb9LQ5gXg0nnl3HL53GEVeBIRygo9FOUO/SamMcbEKa5f57Pq7lpdczeBNi8el4ObLp3NyrMmDGtoxOVwMH5MDjkuZwJ7aYwxw5NVwa3AtNJ81q+uorK8cFjnynU7GV9s49nGmPSTVcFdnOvm+x8+jzz38K6Qi3LdlBV6hnW1bowxyZJVwT1hTO6wQ3tcYQ5j8mw82xiTvrIquIfDFtUYYzKFpRR2E9IYk1lGfXC7nQ4mjskd8ibAxhgz0kZ1cNvMEWNMJhq1wV2Y46K8KMdmjhhjMs6oDO7SAg9j860cqzEmM42q4HaIUF6UM6wl8MYYk2qjJsGsHKsxJluMiuDO90TGs+0mpDEmG2R9cFs5VmNMtsnq4B5XkDOsPSWNMSYdZWVwS/QmZKHdhDTGZKGsSzaHCBPG5JI7zGJTxhiTrrIquAWYODbXao4YY7JaVs2Nc7scFtrGmKyXVcFtk/2MMaNBVgX37mMdXHv/dp7ZfSLVXTHGmKTJquB2OYQTHV7WP/KGhbcxJmtlVXBDZJWk2ync91xtqrtijDFJkXXBDZDndlLf0p3qbhhjTFJkZXD3BEJMKclPdTeMMSYpsi64u/1BAiHlExdXprorxhiTFFm1ACcUViqKcvnExZUsm1+R6u4YY0xSiKom7+Qih4AOIAQEVXWxiHwduBrwAweA61W1NZ62sV5v8eLFumvXroT13xhjRlhcy1FGYqjkUlVd2Ct4nwDOVNWzgb3A7YNoa4wxo96Ij3Gr6jZVDUYfbgemjHQfjDEmkyU7uBXYJiIvicjaPr5+A/DYENsCICJrRWSXiOxqbGxMQJeNMSa9Jfvm5FJVPSoiFcATIrJbVZ8DEJE7gCDwi8G27U1V7wfuh8gYd3LehjHGpI+kXnGr6tHoxxPAw8ASABH5KLAa+JD2c3e0v7bGGDPaJS24RaRARIpOfQ5cAbwuIiuAW4F3qWqfyxv7a5usvhpjTCZJ5lDJeOBhETn1Og+q6lYR2Q/kEBn+ANiuqp8UkUnAA6q6sr+2SeyrMcZkjKQFt6rWAuf08fzsfo4/CqwcqK0xxpgsXPJujDHZzoLbGGMyTFKXvI80EekA9qS6HwlSBpxMdScSJFveS7a8D7D3kq5yVfXMWAdlVZEpYE+2LI8XkV32XtJLtrwPsPeSrkQkrmJLNlRijDEZxoLbGGMyTLYF9/2p7kAC2XtJP9nyPsDeS7qK671k1c1JY4wZDbLtitsYY7KeBbcxxmSYrAhuEfmxiJwQkYwuRCUiU0Xk9yJSIyJviMjNqe7TUIlIrojsEJFXo+/l31Ldp+ESEaeI/FlENqe6L8MhIodE5C8i8kq808/SlYiMFZHfiMju6M/Nhanu01CIyLzov8epP+0i8rl+j8+GMW4RuRjoBP4rnsnr6UpEJgITVfXlaHXEl4B3q2p1irs2aBKpEFagqp0i4gaeB25W1e0p7tqQicgtwGKgWFVXp7o/QxXdz3Wxqmb8ohUR+SnwB1V9QEQ8QH5fe9hmEhFxAkeAt6nq4b6OyYor7ugGC82p7sdwqWqDqr4c/bwDqAEmp7ZXQ6MRndGH7uifjL1KEJEpwCrggVT3xUSISDFwMfAjAFX1Z3poRy0HDvQX2pAlwZ2NRGQGcC7wYoq7MmTRoYVXgBPAE6qase8F+BbwBSCc4n4kQlzbAmaASqAR+El0COuBaP3+TLcGeGigAyy405CIFAL/DXxOVdtT3Z+hUtWQqi4ksiH0EhHJyGEsEVkNnFDVl1LdlwRZqqrnAVcBn4kONWYiF3Ae8H1VPRfoAm5LbZeGJzrc8y7g1wMdZ8GdZqLjwf8N/EJVf5vq/iRC9NfXZ4AVqe3JkC0F3hUdG94EvFNEfp7aLg1dFm0LWA/U9/pN7jdEgjyTXQW8rKrHBzrIgjuNRG/o/QioUdVvpro/wyEi5SIyNvp5HnAZsDulnRoiVb1dVaeo6gwiv8Y+raofTnG3hiSbtgVU1WNAnYjMiz61HMi4G/mnuZYYwySQJdUBReQhYBlQJiL1wL+o6o9S26shWQp8BPhLdGwY4IuquiV1XRqyicBPo3fIHcCvVDWjp9FliWzbFvAm4BfRIYZa4PoU92fIRCQfuBz4RMxjs2E6oDHGjCY2VGKMMRnGgtsYYzKMBbcxxmQYC25jjMkwFtzGGJNhLLjNqCYinac9vk5EvpOq/hgTDwtuY5IgOn/dmKSw4DamHyIyXUSeEpHXoh+nRZ//TxF5b6/jOqMfl0XrqT8I/CVF3TajQFasnDRmGPJ6rVIFKAUeiX7+HSI13n8qIjcA/wG8O8b5lgBnqurBRHfUmFMsuM1o1xOtYAhExriJbJYAcCHwnujnPwO+Fsf5dlhom2SzoRJj4neqPkSQ6M9OtDCYp9cxXSPdKTP6WHAb078/EqkGCPAhItuvARwCFkU/v4bI7j7GjBgLbmP691ngehF5jUjVxlObN/8QuEREdgBvw66yzQiz6oDGGJNh7IrbGGMyjAW3McZkGAtuY4zJMBbcxhiTYSy4jTEmw1hwG2NMhrHgNsaYDPP/AUtAVXM9xknKAAAAAElFTkSuQmCC)

**y = Wx + b**

기울기 W 절편 b에 따라 모양이 정해지기 때문에 x를 넣었을때 y를 구할 수 있습니다.

선형 회귀의 목적도 우리가 가진 데이터를 가장 잘 설명 할 수 있는 1차 함수 W 기울기와 b 절편을 구하는 것 입니다.



### Loss

![평균제곱오차 MSE - 제타위키](https://z-images.s3.amazonaws.com/1/1f/MCEDescendingIntoMLLeft.png)

선형 회귀에서 발생하는 오차, 손실을 Loss 라고 합니다.

* Loss Function MSE ( Mean Squared Error )

  손실을 구할때 가장 널리 쓰이는 방법으로 실제 값 - 예측 데이터를 제곱을 해서 평균을 한 것으로

   ![image-20210819151051481](C:\Users\user\AppData\Roaming\Typora\typora-user-images\image-20210819151051481.png)

  위 사진의 Loss는 0.4 입니다.

  각 오류 값에 제곱을 하는 이유는 마이너스 값 제거, 큰 오류의 Loss를 더 크게 만들어 주기 위함 입니다.

  MSE 말고도 RMSE MAE 등등 많은 손실 함수가 있습니다.

  ![MSE(Mean Squared Error) 간단 정리](https://blog.kakaocdn.net/dn/X23v4/btqAkxvb5U1/PYxp4B4837xRTVGhAdjtFK/img.jpg)

### 경사하강법 Gradient Descent

경사하강법 ( Gradient Descent)는 손실을 최소화 하기 위한 방법 입니다.

![Gradient Descent and Stochastic Gradient Descent - mlxtend](http://rasbt.github.io/mlxtend/user_guide/general_concepts/gradient-optimization_files/ball.png)

파라미터를 임의로 정한 후에 조금씩 변화 시키며 손실을 점점 줄여가는 방법으로 최적의 파라미터를 찾아갑니다.



### 수렴 ( Convergence )

![Linear regression with multiple features - Do It Easy With ScienceProg](https://scienceprog.com/wp-content/uploads/2016/03/cost_function_convergence.png)

선형회귀 분석을 수행하면 기울기와 절편을 변경해 가면서 최적의 값에 수렴 하게 됩니다.

### 학습률 Learning Rate



![Setting the learning rate of your neural network.](https://www.jeremyjordan.me/content/images/2018/02/Screen-Shot-2018-02-24-at-11.47.09-AM.png)

경사하강법 알고리즘은 기울기에 학습률을 곱해서 다음 지점을 결정 합니다.

**학습률이 큰 경우 : 데이터가 무질서 하게 이탈하며, 최저점에 수렴하지 못합니다.**

**학습률이 작은 경우 : 학습시간이 매우 오래 걸리며, 최저점에 도달하지 못합니다.**
![학습률 (Learning rate)](https://img1.daumcdn.net/thumb/R720x0.q80/?scode=mtistory2&fname=http%3A%2F%2Fcfile21.uf.tistory.com%2Fimage%2F99062F4A5D8998492A5DDC)

**low learning rate**: 손실 감소가 선형의 형태를 보이면서 천천히 학습됩니다.

**high learning rate**: 손실 감소가 지수적인 형태를 보이며, 구간에 따라 빠른 학습 혹은 정체가 보입니다.

**very high learning rate**: 매우 높은 학습률은 경우에 따라, 손실을 오히려 증가시키는 상황을 발생시킵니다.

**good learning rate**: 적절한 학습 곡선의ㅇ 형태로, Learning rate를 조절하면서 찾아내야 합니다.



## 로지스틱 회귀 Logistic Regression

![image-20210819171843323](C:\Users\user\AppData\Roaming\Typora\typora-user-images\image-20210819171843323.png)

* **로지스틱 회귀 모델의 필요성**

  Y 값이 카테고리인 즉 연속성이 없어 선형회귀 모델과 다른 방식으로 접근해야 합니다. 

  새로운 관측치가 왔을 때 이를 기존 범주 중 하나로 예측하는 즉 분류를 하는 것 입니다.

* **로지스틱 함수**

  ![image-20210819173820049](C:\Users\user\AppData\Roaming\Typora\typora-user-images\image-20210819173820049.png)

  베르누이 시도에서 1이 나올 확률 μ와 0이 나올 확률 1−μ의 비율(ratio)을 승산비(odds ratio)라고 합니다.

  ![image-20210819174124446](C:\Users\user\AppData\Roaming\Typora\typora-user-images\image-20210819174124446.png)

  위 식인 승산비를 로그 변환한 것이 로지트함수(Logit function)입니다.

  로지트함수의 값은 로그 변환에 의해 **음의 무한대(−∞)부터 양의 무한대 (+∞)**까지의 값을 가질 수 있습니다.

  ![image-20210819174222293](C:\Users\user\AppData\Roaming\Typora\typora-user-images\image-20210819174222293.png)

  로지스틱함수는 로지트함수의 역함수로 **음의 무한대(−∞)부터 양의 무한대 (+∞)까지의 값을 가지는 입력변수를 0부터 1사의 값을 가지는 출력변수로 변환한 것 입니다.**

  

  로지스틱함수도 역시 구해야 할 것은 계수와 절편 즉 가중치(Weight)와 편향(Bias)를 인공 지능 알고리즘이 구하는 것 입니다.
  
  ### 비용 함수 Cost Function
  
  로지스틱 회귀 또한 가중치를 찾아내지만 비용 함수로는 MSE를 사용하지 않습니다.
  
  시그모이드에 MSE로 그래프를 그리면 다음 그림과 같이 로컬 미니멈에 빠질 수 있습니다.
  
  ![img](https://wikidocs.net/images/page/22881/%EB%A1%9C%EC%BB%AC%EB%AF%B8%EB%8B%88%EB%A9%88.PNG)

시그모이드 함수는 0과 1사이의 y값을 반환합니다. 이는 실제 값이 0일때 y값이 1에 가까워지면 오차가 커지며 실제값이 1일 때 y 값이

0에 가까워지면 오차가 커짐을 의미합니다. 그리고 이를 로그 함수로 표현이 가능합니다.

![image-20210819185752630](C:\Users\user\AppData\Roaming\Typora\typora-user-images\image-20210819185752630.png)

![img](https://wikidocs.net/images/page/22881/%EC%86%90%EC%8B%A4%ED%95%A8%EC%88%98.PNG)

의 실제값이 1일 때 −logH(x) 그래프를 사용하고 y의 실제값이 0일 때 −log(1−H(X)) 그래프를 사용해야 합니다. 위의 두 식을 그래프 상으로 표현하면 아래와 같습니다.

![image-20210819190148683](C:\Users\user\AppData\Roaming\Typora\typora-user-images\image-20210819190148683.png)

y가 0이면 ylogH(X)가 없어지고, y가 1이면 (1−y)log(1−H(X))가 없어지는데 이는 각각 y가 1일 때와 y가 0일 때의 앞서 본 식과 같습니다.

![image-20210819190323931](C:\Users\user\AppData\Roaming\Typora\typora-user-images\image-20210819190323931.png)

이때 이 로지스틱 회귀에서 찾아낸 비용 함수를 크로스 엔트로피(Cross Entropy)함수 라고 합니다. 결과적으로 로지스틱 회귀는 크로스 엔트로피 함수를 비용 함수로 사용하고 가중치를 찾기 위해서 크로스 엔트로피의 평균을 취한 함수를 사용합니다.



## 결정 트리 Decision Tree

결정 트리(Decision Tree, 의사결정트리, 의사결정나무)는 분류(Classification)와 회귀(Regression) 모두 가능한 지도 학습 모델 입니다.

결정 트리는 스무 고개 처럼 예/아니오 질문을 이어가면서 학습을 합니다.

![image-20210819223143689](C:\Users\user\AppData\Roaming\Typora\typora-user-images\image-20210819223143689.png)

이렇게 특정 질문에 따라 데이터를 구분하는 모델을 결정 트리 모델이라고 합니다. 한번의 분기 때마다 변수 영역을 두개로 구분합니다.

결정 트리에서 질문이나 정답을 담은 네모 상자를 노드(Node)라고 합니다. 맨 처음 분류 기준을 Root Node라고 하고, 맨 마지막 노드를 

Terminal Node 혹은 Leaf Node 라고 합니다.



![image-20210819223356021](C:\Users\user\AppData\Roaming\Typora\typora-user-images\image-20210819223356021.png)

우선 위와 같이 데이터를 가장 잘 구분 할 수 있는 질문을 기준으로 나눕니다.

![image-20210819223424419](C:\Users\user\AppData\Roaming\Typora\typora-user-images\image-20210819223424419.png)

나뉜 각 범주에서 가장 잘 구분 할수 있는 질문을 기준으로 또 나눕니다.

![image-20210819223501738](C:\Users\user\AppData\Roaming\Typora\typora-user-images\image-20210819223501738.png)

하지만 이것을 지나치게 많이하면 (층이 많이 쌓이면) 보는 것과 같이 Overfitting 이 됩니다.

여기서 하이퍼파라미터(Hyperparameter)를 조정하여 Overfitting을 막을 수 있습니다.



### 엔트로피, 불순도

불순도(Imputity)란 해당 범주 안에 서로 다른 데이터가 얼마나 섞여 있는지를 뜻합니다. 아래 그림에서 위쪽 범주는 불순도가 낮고

아래 범주는 불순도가 높습니다. 

![image-20210819223736757](C:\Users\user\AppData\Roaming\Typora\typora-user-images\image-20210819223736757.png)

한 범주에 하나의 데이터만 있다면 순도가 최고이고 , 한 범주 안에 서로 다른 두 데이터가 정확히 반반 있다면 불순도가 최대가 됩니다. 

결정 트리는 불순도를 최소화 혹은 순도를 최대화 하는 방향으로 학습을 진행합니다.

 

엔트로피 (Entropy)는 불순도(Imputity)를 수치적으로 나타낸 척도입니다. 엔트로피가 높다는 것은 불순도가 높다는 것 입니다.

엔트로피를 구하는 공식은 다음과 같습니다.

![image-20210819224127565](C:\Users\user\AppData\Roaming\Typora\typora-user-images\image-20210819224127565.png)

Pi = 한 영역 안에 존재하는 데이터 가운데 범주 i에 속하는 데이터의 비율



### 정보 획득 Information gain

엔트로피가 1인 상태에서 0.7인 상태로 바꿧다면 정보 획득 (Information gain)은 0.3 입니다. 분기 이전의 엔트로피에서 

분기 이후의 엔트로피를 뺀 수치를 정보 획득이라고 합니다. 정보 획득은 아래와 같이 공식화를 할 수 있습니다.

**Information gain = entropy(parent) - [weighted average] entropy(children)**

entropy(parent)는 분기 이전 엔트로피고 , entropy(children)은 분기 이후 엔트로피 입니다.

이때 weighted average는 가중 평균을 뜻 합니다. 분기 이후 엔트로피에 가중 평균을 하는 이유는 분기를 하면 범주가 2개 이상으로

쪼개지기 때문입니다. 범주가 하나라면 위 엔트로피 공식으로 바로 엔트로피를 구할 수 있지만, 범주가 2개 이상일 경우 가중 평균을

활용하여 분기 이후 엔트로피를 구하는 것입니다.



**결정트리 알고리즘은 정보 획득을 최대화 하는 방향으로 학습이 진행 됩니다.** 어느 feature의 어느 분기점에서 정보 획득이 

최대화되는지 판단을 해서 분기가 진행됩니다.

## 앙상블 Ensemble

앙상블은 조화 또는 통일을 의미합니다.

앙상블 학습은 여러개의 결정 트리(Descision Tree)를 결합하여 하나의 결정 트리보다 더 좋은 성능을 내는 머신러닝 기법입니다.

앙상블 학습의 핵심은 여러 개의  약 분류기 (Weak Classifier) 를 결합하여 강 분류기 (Strong Classifier)를 만드는 것 입니다.

그리하여 모델의 정확성이 향상됩니다.

* 앙상블 학습 유형

  앙상블 학습은 일반적으로 보팅(Voting), 배깅(Bagging), 부스팅(Boosting) 세 가지의 유형으로 나눌 수 있습니다.

  * 보팅(Voting)
    * 하드 보팅(Hard Voting)
      * 다수의 분류기가 예측한 결과값을 최종 결과로 선정
      * ![앙상블(Ensemble) 기법](https://media.vlpt.us/images/fiifa92/post/452ed087-73fb-49af-936f-9d0234b6070f/%ED%95%98%EB%93%9C%EB%B3%B4%ED%8C%85.png)
    * 소프트 보팅(Soft Voting)
      * 모든 분류기가 예측한 레이블 값의 결정 확률 평균을 구한 뒤 가장 확률이 높은 레이블 값을 최종 결과로 선정
      * ![앙상블(Ensemble) 기법](https://media.vlpt.us/images/fiifa92/post/34ff89d6-2da6-47cb-9067-e0629a63d429/%EC%86%8C%ED%94%84%ED%8A%B8%20%EB%B3%B4%ED%8C%85.png)

  * 배깅(Bagging)

    * ![머신러닝 - 11. 앙상블 학습 (Ensemble Learning): 배깅(Bagging)과 부스팅(Boosting)](https://blog.kakaocdn.net/dn/b4wG8O/btqyfYW98AS/YZBtUJy3jZLyuik1R0aGNk/img.png)
    * **배깅(Bagging)은 Bootstrap Aggregating의 줄임말로, 부트스트래핑을 이용한 앙상블 학습법**
    * 부트스트래핑과 패이스팅
      - 부트스트래핑 : 학습 데이터셋에서 중복을 허용하여 랜덤하게 추출하는 방식 (리샘플링)
      - 페이스팅 : 학습 데이터셋에서 중복 없이 랜덤하게 추출하는 방식
    * 부트스트래핑 장단점
      - 장점: 분산 감소
      - 단점 : 중복으로 인해, 특정 샘플은 사용되지 않고 특정 샘플은 여러번 사용되어 편향될 가능성
        - OOB(Out-of-Bag) 샘플: 샘플링 되지 않은 나머지 샘플

    * 배깅을 적용한 대표적인 기법으로 Random Forest 가 있습니다.

  * 부스팅(Boosting)

    * 부스팅은 가중치를 활용하여 약 분류기를 강 분류기로 만드는 방법입니다. 

      부스팅은 모델 간 팀워크가 이루어집니다. 처음 모델이 예측을 하면 그 예측 결과에 따라 데이터에 가중치가 부여되고,

      부여된 가중치가 다음 모델에 영향을 줍니다. 잘못 분류된 데이터에 집중하여 새로운 분류 규칙을 
    
      만드는 단계를 반복합니다.
    
    * ![img](https://blog.kakaocdn.net/dn/kCejr/btqyghvqEZB/9o3rKTEsuSIDHEfelYFJlk/img.png)
    
    * 장단점
      - **장점**: 오답에 대해 높은 가중치를 부여하고 정답에 대해 낮은 가중치를 부여하여 오답에 더욱 집중
        - **단점**: 이상치(Outlier)에 취약
      
    * 부스팅의 대표적인 알고리즘
    
      * AdaBoost
      * Gradient Boost Machine
      * XGBoost
      * LightBGM

## 랜덤 포레스트 Random Forest

랜덤 포레스트의 포레스트는 숲(Forest)입니다. 결정 트리의 트리는 나무(Tree)입니다.

이름 처럼 나무가 모여 숲을 이룹니다. 즉 결정 트리(Decision Tree)가 모여 랜덤 포레스트(Random Forest)를 구성합니다.

결정트리 하나만으로 머신러닝을 할 수 있지만, 훈련 데이터에 Overfitting 되는 경향이 있습니다.

하지만 이 랜덤 포레스트(Random Forest)에선 여러개의 결정 트리(Decision Tree)를 사용해 Overfitting 되는

단점을 해결 할 수 있습니다.



Feature가 많아지면 많아질수록 트리의 가지가 많아질 것 이고 이는 Overfitting이 될 수 있습니다.

하지만 여러개의 Feature중 몇개의 Feature만 선택을 하여 결정트리들을 만들면 이 예측 값들을 모와 과적합을 방지하면서

결과 값을 예측 할 수 있습니다. 

이렇게 의견을 통합해서 다수결의 원칙으로 Ensemble을 합니다.

**문제를 풀 때도 한명의 똑똑한 사람보다 100명의 평범한 사람이 더 잘 푸는 원리입니다.**
![image-20210819225520070](C:\Users\user\AppData\Roaming\Typora\typora-user-images\image-20210819225520070.png)

기존 각 결정 트리(Decision Tree)의 경계는 다소 모호하고 Overfitting이 된 모습을 볼 수 있는데

5개의 결정 트리(Decision Tree)를 기반으로 평균을 내어서 만든 랜덤 포레스트는 비교적 깔끔한 것을 볼 수 있습니다.

## 에이다 부스트 AdaBoost

에이다 부스트는 부스팅 기법 중 가장 기본이 되는 알고리즘입니다.



아래와 같이 노드 하나에 두개의 리프(Leaf)를 가지는 트리를 stump라고 합니다.

![image-20210819215922023](C:\Users\user\AppData\Roaming\Typora\typora-user-images\image-20210819215922023.png)

AdaBoost는 아래와 같이 여러 개의 stump로 구성이 되어있습니다. 이를 Forest of stumps라고 합니다.

![image-20210819220002848](C:\Users\user\AppData\Roaming\Typora\typora-user-images\image-20210819220002848.png)

Random Forest는 여러개의 트리를 합산해 최종 결과는 도출 합니다. 다수결의 원칙 처럼 각 최종분류를 하는데 있어 모두

동등한 가중치를 가지고 있습니다. 

하지만 AdaBoost 에선 각각의 가중치가 다릅니다. 여기서 가중치가 높다는 것을 Amount of Say가 높다고 표현을 합니다.

![image-20210819220345382](C:\Users\user\AppData\Roaming\Typora\typora-user-images\image-20210819220345382.png)



AdaBoost는 3가지의 특징을 가지고 있습니다.

 * 약한 학습기(Weak Learner)으로 구성되어 있고 Stump의 형태 입니다.
 * 어떤 Stump는 다른 Stump 보다 가중치가 높습니다.
 * 각 Stump의 Error은 다음 Stump의 결과에 영향을 줍니다.



### AdaBoost 작동 원리

![image-20210819221139788](C:\Users\user\AppData\Roaming\Typora\typora-user-images\image-20210819221139788.png)

Chest Pain, Blocked Arteries, Patient Weight에 따른 Heart Disease 여부에 대한 데이터입니다. 맨 처음 Sample Weight는 8개의 

데이터 모두 동일하게 1/8입니다.



각 Stump를 만든 후 지니 계수를 구합니다.

![image-20210819221345580](C:\Users\user\AppData\Roaming\Typora\typora-user-images\image-20210819221345580.png)



마지막 Stump의 지니 계수가 가장 작기 때문에 Forest의 첫 Stump로 지정합니다.

### Amount Of Say 구하기

틀리게 분류한 것이 No Heart Disease의 Incorrect로 1개밖에 없습니다. 따라서 Total Error = 1/8입니다.

![image-20210819221657766](C:\Users\user\AppData\Roaming\Typora\typora-user-images\image-20210819221657766.png)

모든 Sample Weight의 합은 1이기 때문에, Total Error은 0과 1사이의 값을 갖습니다. Total Error가 Amount of Say를 결정하고. 

Amount of Say는 최종 분류에 있어 해당 Stump가 얼마나 영향을 주는지를 뜻합니다. 



* Amount of Say 공식

![image-20210819221857800](C:\Users\user\AppData\Roaming\Typora\typora-user-images\image-20210819221857800.png)

Amount of Say를 그래프로 그리면 아래와 같습니다.

![img](https://blog.kakaocdn.net/dn/c2v7ec/btqyMrRViSc/AzZqB4fpkqXGqmfe2ciebK/img.png)

X 축은 Total Error Y 축은 Amount of Say 입니다 Total Error가 0이면 굉장히 큰 양수이고

Total Error가 1이면 굉장히 작은 음수가 됩니다. 따라서 Total Error가 0이면 분류를 올바르게 했다는 뜻이고

1이면 분류를 반대로 한다는 뜻 입니다.



위 Stump 에서 Total Error = 1/8이라고 했으니

![image-20210819222143253](C:\Users\user\AppData\Roaming\Typora\typora-user-images\image-20210819222143253.png

![img](https://blog.kakaocdn.net/dn/RBEZr/btqyKIAiXSG/gr6jtUzKTub3udRFVLIrG0/img.png)

Amount of Say 가 0.98인 지점 입니다.



### 샘플 가중치

Adaboost에서는 하나의 Stump가 잘못 분류한 sample에 대해서는 다음 Stump로 넘겨줄 때 가중치를 더 높여서 넘겨줍니다. 그래야 다음 Stump에서 해당 Sample에 더 집중해서 올바로 분류해주기 때문입니다.

다시 맨 처음에 했던 방식 대로 계속 반복을 하면 됩니다.



### 최종 분류

여러 차례 진행 하다보면 아래와 같이 각 Stump 마다의 Amount of Say 가 나오게 됩니다.

왼쪽은 Heart Disease가 있다고 한 Stump이고 오른쪽은 없다고 판단한 Stump 입니다.

결국 Has Heart Disease의 Total 이 Does Not Have Heart Disease 보다 크니 Has Heart Disease 라고 예측 할 수 있게 됩니다.

![image-20210819222505829](C:\Users\user\AppData\Roaming\Typora\typora-user-images\image-20210819222505829.png)

각 Stump 하나하나의 분류력은 굉장히 약하지만 여러 Stump의 결과를 종합하면 강한 학습기(Strong Learner)가 됩니다.

## XGBoost

## LightGBM