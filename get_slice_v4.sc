/* get_slice_v4.sc - 支持过程间语义扩展（优化版） */
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

  val files = Option(dir.listFiles)
    .getOrElse(Array.empty)
    .filter(f => f.isFile && f.getName.endsWith(".c"))

  val maxDepth = 10
  val maxMethodSpan = 400

  // === 敏感函数集合 ===
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
    var cpg: Cpg = null
    try {
      cpg = importCode.c(inputPath = file.getAbsolutePath)

      // 1. 查找 Sinks
      val apiSinks = cpg.call.filter(node => sensitiveFuncs.contains(node.name)).l
      val opSinks = cpg.call.filter(node => arrayOps.contains(node.name) || pointerOps.contains(node.name)).l
      val allSinks = (apiSinks ++ opSinks).distinct

      if (allSinks.nonEmpty) {
        // 2. 逆向数据依赖
        val dataDepNodes = allSinks.argument.repeat(_.ddgIn)(_.maxDepth(maxDepth).emit).l

        // 3. 逆向控制依赖
        val controlDepNodes = allSinks.repeat(_.cdgIn)(_.maxDepth(maxDepth).emit).l

        // 4. 提取控制流条件行
        val conditionLines = controlDepNodes.flatMap {
          case node: ControlStructure => node.lineNumber
          case node: AstNode => node.lineNumber
          case _ => None
        }.filter(_ > 0)

        // 5. 方法包裹
        val methodNodes = (allSinks ++ dataDepNodes ++ controlDepNodes).flatMap(_.start.method.l).distinct

        // 6. 聚合初步切片节点
        val allSliceNodes = (allSinks ++ dataDepNodes ++ controlDepNodes).distinct

        // 7. 过程间扩展 (Callee Expansion)
        val callsInSlice = allSliceNodes.collect { case c: Call => c }
        val calleeMethods = callsInSlice.callee.l.distinct
        val extensionMethods = calleeMethods.filterNot(m => methodNodes.contains(m))

        val extensionLines = extensionMethods.flatMap { m =>
          val start = m.lineNumber.getOrElse(0)
          val end = m.lineNumberEnd.getOrElse(0)
          val span = end - start + 1
          if (start > 0 && end >= start && span <= maxMethodSpan) (start to end).toList else List.empty[Int]
        }

        // 8. 提取行号 (合并所有来源)
        val flowLines = allSliceNodes.flatMap(_.lineNumber).filter(_ > 0)
        val methodHeaderLines = methodNodes.flatMap(_.lineNumber).filter(_ > 0)
        val methodFooterLines = methodNodes.flatMap(_.lineNumberEnd).filter(_ > 0)
        val paramLines = methodNodes.flatMap(_.parameter.lineNumber).filter(_ > 0)

        val finalLineSet = (flowLines ++ conditionLines ++ methodHeaderLines ++ methodFooterLines ++ paramLines ++ extensionLines)
          .distinct
          .sorted

        val finalLinesStr = finalLineSet.mkString(",")
        println(s"###RESULT###:${file.getName}:$finalLinesStr")
      } else {
        println(s"###RESULT###:${file.getName}:")
      }
    } catch {
      case e: Exception =>
        System.err.println(s"Error processing ${file.getName}: ${e.getMessage}")
        println(s"###RESULT###:${file.getName}:")
    } finally {
      if (cpg != null) {
        try cpg.close()
        catch {
          case closeEx: Exception =>
            System.err.println(s"Warning: failed to close CPG for ${file.getName}: ${closeEx.getMessage}")
        }
      }
    }
  }
}
