import copy

# 가변형 변수(e.g. list, class) 같은 경우에는 등호를 이용하여 값을 복사할 경우 얕은 복사가 된다.
# 즉, 주소 값을 공유하게 되어 하나의 값을 바꾸게 될 경우 다른 값도 변화 하게 된다.
l1 = [1, 2, 3]
l2 = l1
l2[0] = 3
print("="*30)
print("l1 주소값: "+str(l1.id())+",  l2 주소값"+str(l2.id()))
print("값 변경 전: l1 = [1, 2, 3],  l2=[1, 2, 3]")
print("값 변경 후: l1 = "+str(l1)+",  l2="+str(l2))

# 이를 깊은 복사(deepcopy)를 통하여 해결 할 수 있다.
# 요약하자면, 얕은 복사 = 주소 값 복사, 깊은 복사 = 값 복사라고 볼 수 있다
l3 = [1, 2, 3]
l4 = copy.deepcopy(l3)
l4[0] = 3
print("="*30)
print("l3 주소값: "+str(l3.id())+",  l4 주소값"+str(l4.id()))
print("값 변경 전: l3 = [1, 2, 3],  l4=[1, 2, 3]")
print("값 변경 후: l3 = "+str(l3)+",  l4="+str(l4))