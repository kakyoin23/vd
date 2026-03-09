/* get_slice_v5_plus.sc - 支持过程间语义扩展 */
import io.shiftleft.semanticcpg.language._
import io.shiftleft.codepropertygraph.generated.Cpg
import io.shiftleft.codepropertygraph.generated.nodes._
import java.io.File

@main def main(inputDir: String): Unit = {
  val dir = new File(inputDir)
  if (!dir.exists || !dir.isDirectory) {
    println(s"Error: $inputDir is not a valid directory")
    return
  }

  val files = dir.listFiles.filter(f => f.isFile && f.getName.endsWith(".c"))

  // === 敏感函数集合 (保持不变) ===
  val sensitiveFuncs = Set(
    "memcpy", "memmove", "memset", "bcopy", "strcpy", "strncpy", "strcat", "strncat", 
    "sprintf", "snprintf", "vsprintf", "vsnprintf", "wcscpy", "wcsncpy",
    "strlen", "wcslen", 
    "malloc", "free", "realloc", "calloc", "alloca", "new", "delete", "delete[]",
    "system", "popen", "pclose", "execl", "execlp", "execle", "execv", "execvp", "execvpe",
    "CreateProcess", "ShellExecute",
    "scanf", "fscanf", "sscanf", "vscanf", "vsscanf", "gets", "fgets", "getchar", "fgetc",
    "read", "recv", "recvfrom", "fread", "printf", "fprintf", "syslog", "dprintf", "getenv",
    "mkstemp", "mktemp", "tmpnam", "tempnam", "access", "stat", "lstat", "open", "chmod", "chown", "rename", "unlink", "remove",
    "dlopen", "LoadLibrary", "rand", "srand", "random", "atoi", "atol", "atoll"
  )

  val arrayOps = Set("<operator>.indexAccess", "<operator>.indirectIndexAccess")
  val pointerOps = Set("<operator>.indirection", "<operator>.fieldAccess", "<operator>.indirectFieldAccess")

  files.foreach { file =>
    try {
      val cpg = importCode.c(inputPath = file.getAbsolutePath)
      
      // 1. 查找 Sinks
      val apiSinks = cpg.call.filter(node => sensitiveFuncs.contains(node.name))
      val opSinks = cpg.call.filter(node => arrayOps.contains(node.name) || pointerOps.contains(node.name))
      val allSinks = apiSinks ++ opSinks

      if (allSinks.nonEmpty) {
         // === 2. 逆向数据依赖 ===
         val dataDepNodes = allSinks.argument.repeat(_.ddgIn)(_.maxDepth(10).emit).l

         // === 3. 逆向控制依赖 ===
         val controlDepNodes = allSinks.repeat(_.cdgIn)(_.maxDepth(10).emit).l

         // === 4. 提取控制流条件行 ===
         val conditionLines = controlDepNodes.flatMap { 
             case node: ControlStructure => node.lineNumber 
             case node: AstNode => node.lineNumber 
             case _ => None
         }.filter(_ > 0)

         // === 5. 方法包裹 (主要漏洞所在的函数) ===
         val methodNodes = (allSinks.l ++ dataDepNodes ++ controlDepNodes).flatMap(_.start.method.l).distinct

         // === 6. 聚合初步切片节点 ===
         val allSliceNodes = (allSinks.l ++ dataDepNodes ++ controlDepNodes).distinct

         // =========================================================
         // === [新增关键逻辑] 7. 过程间扩展 (Callee Expansion) ===
         // =========================================================
         // 逻辑：如果在切片中发现调用了其他函数，且该函数在当前文件中定义，
         // 则将该被调函数的所有代码行也加入切片。
         
         // 1. 找出切片中所有的 CALL 节点
         val callsInSlice = allSliceNodes.collect { case c: Call => c }
         
         // 2. 找到这些 Call 对应的 Method 定义 (Callee)
         // 注意：.callee 只能找到当前 CPG 中存在的函数。
         // 如果是 printf 等库函数，CPG 里没有 Method体，会自动忽略，这正是我们要的。
         val calleeMethods = callsInSlice.callee.l.distinct
         
         // 3. 排除掉自己调用自己的递归情况 (可选，防止冗余)
         // 也可以排除掉 main 函数 (通常辅助函数不是 main)
         val extensionMethods = calleeMethods.filterNot(m => methodNodes.contains(m))

         // 4. 获取被调函数的全部行
         // 这里我们策略比较激进：如果辅助函数被调用了，就保留它的全貌，
         // 这样能最大程度保留语义。
         val extensionLines = extensionMethods.flatMap { m =>
             val start = m.lineNumber.getOrElse(0)
             val end = m.lineNumberEnd.getOrElse(0)
             if (start > 0 && end >= start) (start to end).toList else List()
         }

         // =========================================================
         
         // === 8. 提取行号 (合并所有来源) ===
         
         // A. 原始切片的数据流/控制流行
         val flowLines = allSliceNodes.flatMap(_.lineNumber).filter(_ > 0)

         // B. 主函数头尾
         val methodHeaderLines = methodNodes.flatMap(_.lineNumber).filter(_ > 0)
         val methodFooterLines = methodNodes.flatMap(_.lineNumberEnd).filter(_ > 0)

         // C. 主函数参数行
         val paramLines = methodNodes.flatMap(_.parameter.lineNumber).filter(_ > 0)

         // D. [新增] 扩展的被调函数行
         val finalLineSet = (flowLines ++ conditionLines ++ methodHeaderLines ++ methodFooterLines ++ paramLines ++ extensionLines).distinct.sorted

         val finalLinesStr = finalLineSet.mkString(",")
         
         if (finalLinesStr.nonEmpty) {
             println(s"###RESULT###:${file.getName}:$finalLinesStr")
         } else {
             println(s"###RESULT###:${file.getName}:")
         }

      } else {
         println(s"###RESULT###:${file.getName}:") 
      }
      
      cpg.close()
    } catch {
      case e: Exception => System.err.println(s"Error processing ${file.getName}: ${e.getMessage}")
    }
  }
}