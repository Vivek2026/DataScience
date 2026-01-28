class Node:
    def __init__(self, data):
        self.data = data
        self.next = None


class LinkedList:
    def __init__(self):
        self.head = None

    def add_node(self, data):
        """Add a node to the end of the list."""
        new_node = Node(data)
        if not self.head:
            self.head = new_node
            return
        current = self.head
        while current.next:
            current = current.next
        current.next = new_node

    def print_list(self):
        """Print the linked list."""
        if not self.head:
            print("List is empty.")
            return
        current = self.head
        while current:
            print(current.data, end=" -> ")
            current = current.next
        print("None")

    def delete_nth_node(self, n):
        """Delete the nth node (1-based index)."""
        if not self.head:
            raise Exception("Cannot delete from an empty list.")

        if n <= 0:
            raise ValueError("Index must be 1 or greater.")

        # Deleting the head
        if n == 1:
            deleted_data = self.head.data
            self.head = self.head.next
            print(f"Deleted node with data: {deleted_data}")
            return

        current = self.head
        count = 1

        # Traverse to the (n-1)th node
        while current and count < n - 1:
            current = current.next
            count += 1

        if not current or not current.next:
            raise IndexError("Index out of range.")

        deleted_data = current.next.data
        current.next = current.next.next
        print(f"Deleted node with data: {deleted_data}")


if __name__ == "__main__":
    ll = LinkedList()

    print("Adding nodes:")
    for value in [10, 20, 30, 40, 50]:
        ll.add_node(value)
    ll.print_list()

    print("\nDeleting 3rd node:")
    ll.delete_nth_node(3)
    ll.print_list()

    print("\nDeleting 1st node (head):")
    ll.delete_nth_node(1)
    ll.print_list()

    print("\nAttempting to delete node at index 10:")
    try:
        ll.delete_nth_node(10)
    except Exception as e:
        print("Error:", e)

    print("\nDeleting all remaining nodes:")
    try:
        while True:
            ll.delete_nth_node(1)
    except Exception as e:
        print("Error:", e)
