from typing import List


class Solution:
    def numUniqueEmails(self, emails: List[str]) -> int:
        output = {}
        ans = 0
        for email in emails:
            local_name, hostname = email.split('@')
            local_name = local_name.replace('.', '')
            local_name = local_name.split('+')[0]
            simplified_mail = local_name + '@' +hostname
            if simplified_mail in output:
                pass
            else:
                output[simplified_mail] = True
                ans = ans + 1

        return ans


s = Solution()

print(s.numUniqueEmails(
    ["test.email+alex@leetcode.com", "test.email.leet+alex@code.com"])
)
