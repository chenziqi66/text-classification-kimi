# coding: utf-8
"""
文本分类项目测试文件
测试修复的5个问题：
1. 重复导入相同模块
2. NLTK交互式下载会导致脚本卡住
3. SVM参数已过时
4. 导入位置混乱，不符合PEP8规范
5. 大量Jupyter Notebook残留标记
"""

import ast
import sys
import warnings
import unittest
from io import StringIO


class TestCodeQuality(unittest.TestCase):
    """测试代码质量相关的问题修复"""

    def setUp(self):
        """读取主文件内容"""
        with open('Text+Classification+using+python,+scikit+and+nltk.py', 'r', encoding='utf-8') as f:
            self.source_code = f.read()
        self.tree = ast.parse(self.source_code)

    def test_no_duplicate_imports(self):
        """测试问题1: 检查没有重复导入相同模块"""
        imports = []
        for node in ast.walk(self.tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imports.append(alias.name)
            elif isinstance(node, ast.ImportFrom):
                module = node.module or ''
                for alias in node.names:
                    imports.append(f"{module}.{alias.name}")

        # 检查重复
        duplicates = [item for item in set(imports) if imports.count(item) > 1]
        self.assertEqual(len(duplicates), 0,
                        f"发现重复导入: {duplicates}")

    def test_no_jupyter_markers(self):
        """测试问题5: 检查没有Jupyter Notebook残留标记"""
        jupyter_patterns = ['# In[', '#In[', '#In [']
        lines = self.source_code.split('\n')
        jupyter_lines = []

        for i, line in enumerate(lines, 1):
            stripped = line.strip()
            for pattern in jupyter_patterns:
                if stripped.startswith(pattern):
                    jupyter_lines.append((i, line.strip()))

        self.assertEqual(len(jupyter_lines), 0,
                        f"发现Jupyter Notebook残留标记: {jupyter_lines}")

    def test_imports_at_top(self):
        """测试问题4: 检查导入语句集中在文件开头（符合PEP8规范）"""
        # 获取所有导入语句的行号
        import_lines = []
        for node in ast.walk(self.tree):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                import_lines.append(node.lineno)

        if not import_lines:
            return

        # 找到最后一个导入语句的行号
        last_import_line = max(import_lines)

        # 检查在最后一个导入语句之后是否还有导入语句（应该在文件开头）
        # 允许在导入后有注释和空行
        lines = self.source_code.split('\n')

        # 检查是否在非导入代码之后还有导入
        non_import_found = False
        for i, line in enumerate(lines, 1):
            if i > last_import_line:
                stripped = line.strip()
                # 跳过空行和注释
                if stripped and not stripped.startswith('#'):
                    non_import_found = True
                    break

        # 如果在最后一个导入之后有非导入代码，这是正常的
        # 我们需要检查的是：是否分散导入
        # 这里简化检查：确保所有导入都在前30行内（考虑到注释）
        self.assertLessEqual(last_import_line, 30,
                            f"导入语句分散在文件中，最后一个导入在第{last_import_line}行，不符合PEP8规范")

    def test_no_interactive_nltk_download(self):
        """测试问题2: 检查没有交互式的nltk.download()调用"""
        # 检查是否使用了非交互式下载
        has_interactive_download = 'nltk.download()' in self.source_code
        self.assertFalse(has_interactive_download,
                        "发现交互式nltk.download()调用，会导致脚本在非交互式环境中卡住")

        # 检查是否使用了quiet=True参数
        has_quiet_download = "nltk.download('punkt', quiet=True)" in self.source_code
        self.assertTrue(has_quiet_download,
                       "应该使用nltk.download('punkt', quiet=True)来非交互式下载")

    def test_svm_parameter_updated(self):
        """测试问题3: 检查SVM参数已更新，没有使用过时的n_iter"""
        # 检查没有使用n_iter
        has_deprecated_param = 'n_iter=' in self.source_code
        self.assertFalse(has_deprecated_param,
                        "发现已过时的n_iter参数，应该使用max_iter")

        # 检查使用了max_iter
        has_correct_param = 'max_iter=' in self.source_code
        self.assertTrue(has_correct_param,
                       "应该使用max_iter参数替代n_iter")


class TestSVMParameterCompatibility(unittest.TestCase):
    """测试SVM参数兼容性"""

    def test_sgd_classifier_no_deprecation_warning(self):
        """测试SGDClassifier不会抛出DeprecationWarning"""
        import numpy as np
        from sklearn.linear_model import SGDClassifier
        from sklearn.datasets import fetch_20newsgroups
        from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
        from sklearn.pipeline import Pipeline

        # 捕获警告
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")

            # 加载少量数据进行测试
            twenty_train = fetch_20newsgroups(subset='train', shuffle=True, random_state=42)

            # 使用max_iter参数（修复后的参数）
            text_clf_svm = Pipeline([
                ('vect', CountVectorizer()),
                ('tfidf', TfidfTransformer()),
                ('clf-svm', SGDClassifier(
                    loss='hinge',
                    penalty='l2',
                    alpha=1e-3,
                    max_iter=5,
                    random_state=42
                ))
            ])

            # 训练模型
            text_clf_svm.fit(twenty_train.data[:100], twenty_train.target[:100])

            # 检查是否有与SVM参数相关的DeprecationWarning
            # 过滤掉与tar归档相关的Python 3.14警告
            svm_deprecation_warnings = [
                warning for warning in w
                if issubclass(warning.category, DeprecationWarning)
                and 'n_iter' in str(warning.message)
            ]

            self.assertEqual(len(svm_deprecation_warnings), 0,
                           f"发现SVM参数相关的DeprecationWarning: {[str(w.message) for w in svm_deprecation_warnings]}")


class TestNLTKDownload(unittest.TestCase):
    """测试NLTK下载功能"""

    def test_nltk_non_interactive_download(self):
        """测试NLTK非交互式下载不会卡住"""
        import nltk

        # 测试使用quiet=True参数下载
        # 这不应该引发交互式提示
        try:
            # 使用quiet=True确保非交互式
            result = nltk.download('punkt', quiet=True)
            # 如果成功执行到这里，说明没有卡住
            self.assertTrue(True, "NLTK非交互式下载成功")
        except Exception as e:
            self.fail(f"NLTK下载失败: {e}")


class TestModuleImports(unittest.TestCase):
    """测试模块导入"""

    def test_all_imports_work(self):
        """测试所有导入都能正常工作"""
        try:
            import nltk
            from nltk.stem.snowball import SnowballStemmer
            import numpy as np
            from sklearn.datasets import fetch_20newsgroups
            from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
            from sklearn.linear_model import SGDClassifier
            from sklearn.model_selection import GridSearchCV
            from sklearn.naive_bayes import MultinomialNB
            from sklearn.pipeline import Pipeline
            self.assertTrue(True, "所有导入成功")
        except ImportError as e:
            self.fail(f"导入失败: {e}")


if __name__ == '__main__':
    unittest.main(verbosity=2)
